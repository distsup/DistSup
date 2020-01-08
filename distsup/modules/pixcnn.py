# -*- coding: utf8 -*-
#   Copyright 2019 JSALT2019 Distant Supervision Team
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from distsup.modules import quantizers
from distsup import utils
from distsup.utils import safe_squeeze


class ConditioningAdder(nn.Module):
    """Add conditioning to every frame of a 3D tensor or pixel of a 4D one.

    Args:
        cond_channels: list of dicts with keys (cond_dim, reduction_factor)
    """
    def __init__(self, cond_channels, out_channels, **kwargs):
        super(ConditioningAdder, self).__init__(**kwargs)
        self.cond_convs = nn.ModuleList([
            nn.Conv1d(cc['cond_dim'], out_channels, 1, bias=False)
            for cc in cond_channels])

    def forward(self, x, conds):
        """x is BS x C x T x H!!!
           each c in conds is BS x T' x 1 x C'
        """
        assert len(conds) == len(self.cond_convs)
        bs, c, t = x.shape[:3]

        for cconv, cond in zip(self.cond_convs, conds):
            c_bs, c_t, c_h, c_c = cond.size()
            cond = safe_squeeze(cond, 2).permute(0, 2, 1)
            cond = cconv(cond)
            # expand cond to length of x
            cond = cond.repeat_interleave(t // c_t, 2)
            if x.dim() == 4:
                cond = cond.unsqueeze(3)
            x = x + cond
        return x


class GatedMaskedStack2D(nn.Module):
    """A gated conv split into the H/V towers, as described in [1]

    This module takes a pair of inputs (the previous vertical and
    horizontal one) and produces two outputs.

    Each pixel is conditioned on all pixels in the upper rows,
    and directly to the left.

    [1] https://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders.pdf
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 cond_channels=(), dilation=1, bias_shape=1, mask_self=True,
                 **kwargs):
        super(GatedMaskedStack2D, self).__init__(**kwargs)
        assert kernel_size % 2 == 1
        K2 = kernel_size // 2 + 1
        K = kernel_size
        self.dilation = _pair(dilation)
        bias_shape = _pair(bias_shape)

        self.v_kernel = nn.Parameter(torch.Tensor(
            out_channels * 2, in_channels, K2, K))
        self.register_buffer('v_mask',
                             torch.ones(self.v_kernel.size()))
        self.v_bias = nn.Parameter(torch.zeros(1, out_channels*2, *bias_shape))
        if mask_self in [True, 'v', 'hv']:
            self.v_mask[:, :, -1, :] = 0

        self.h_kernel = nn.Parameter(torch.Tensor(
            out_channels * 2, in_channels, 1, K2))
        self.register_buffer('h_mask',
                             torch.ones(self.h_kernel.size()))
        self.h_bias = nn.Parameter(torch.zeros(1, out_channels*2, *bias_shape))
        if mask_self in [True, 'h', 'hv']:
            self.h_mask[:, :, -1, -1] = 0

        nn.init.kaiming_uniform_(self.v_kernel, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.h_kernel, a=math.sqrt(5))

        self.v_to_h = nn.Conv2d(2 * out_channels, 2 * out_channels, 1,
                                bias=False)

        if not mask_self:
            self.h_to_h_res = nn.Conv2d(out_channels, out_channels, 1,
                                        bias=False)
        else:
            self.h_to_h_res = None

        self.cond_to_v = ConditioningAdder(cond_channels, out_channels * 2)
        self.cond_to_h = ConditioningAdder(cond_channels, out_channels * 2)

    def extra_repr(self):
        return f'dilation={self.dilation}'

    def forward(self, v, h, cond=()):
        """
        h,v size is Batch x Channels x Time/Height x Freqs/Width
        """
        K2 = self.v_kernel.size(-1) // 2
        dil_w, dil_h = self.dilation
        # if dil_h > 1 or dil_w > 1:
        #     breakpoint()
        # V stack
        # Pad heigth from top
        # Pad width from both sides
        v = F.pad(v, (K2 * dil_h, K2 * dil_h, K2 * dil_w, 0))
        v = F.conv2d(v, self.v_kernel * self.v_mask, dilation=self.dilation)

        # H stack
        # Pad only from the right
        h_orig = h
        h = F.pad(h, (K2 * dil_h, 0, 0, 0))
        h = F.conv2d(h, self.h_kernel * self.h_mask, dilation=self.dilation)

        # Add the contribution from v to h
        h = h + self.v_to_h(v) + self.h_bias
        h = self.cond_to_h(h, cond)

        # Once h info is branched, aply conditioning and gating to v

        v = v + self.v_bias
        v = self.cond_to_v(v, cond)
        gv1, gv2 = v.chunk(2, dim=1)
        v = torch.sigmoid(gv1) * torch.tanh(gv2)

        # Now apply gate and residual connection to h
        gh1, gh2 = h.chunk(2, dim=1)
        h = torch.sigmoid(gh1) * torch.tanh(gh2)
        to_skip = h
        if self.h_to_h_res is not None:
            h = h_orig + self.h_to_h_res(h)

        return v, h, to_skip


class MaskedConv2D(nn.Conv2d):
    """
    A convolution that masks pixels which ignores pixel below and to the right.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 mask_self=True, bias=True, **kwargs):
        assert kernel_size % 2 == 1
        K2 = kernel_size // 2 + 1
        K = kernel_size
        super(MaskedConv2D, self).__init__(
            in_channels, out_channels, (K2, K),
            bias=bias, stride=1, padding=0, dilation=dilation, **kwargs)
        mask = torch.ones(self.weight.shape)
        if mask_self:
            # The 'A' mask forbids conecting a pixel to itself
            mask[:, :, K2 - 1, K2 - 1:] = 0
        else:
            # The 'B' mask forbids looking into the future pixels, but allows
            # The self-connection
            mask[:, :, K2 - 1, K2:] = 0
        mask[:, :, K2:, :] = 0

        self.register_buffer('mask', mask)

    def forward(self, input):
        """
        Input is N x C x H x W
        """
        dil_h, dil_w = self.dilation
        pad_h = (self.weight.size(-2) - 1) * dil_h
        pad_w = (self.weight.size(-1) // 2) * dil_w
        input = F.pad(input, [pad_w, pad_w, pad_h, 0])
        return F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BasePixelCNN(nn.Module):
    """A base PixelCNN which assumes SoftMax outputs for all pixels.
    """
    def __init__(self,
                 reconstruction_channel=None,
                 quantizer=dict(
                     class_name=quantizers.SoftmaxUniformQuantizer,
                     num_levels=4
                     ),
                 **kwargs):
        super(BasePixelCNN, self).__init__(**kwargs)
        self.quantizer = utils.construct_from_kwargs(quantizer)
        self.reconstruction_channel = reconstruction_channel

    def get_inputs_and_targets(self, x, x_lens=None):
        """Quantize and permute inputs.

        x: tensor of shape BS x T x H x 1, normalized to 0-1 range.
        """
        del x_lens  # unused
        if self.reconstruction_channel is None:
            assert x.size(3) == 1
        else:
            x = x[:, :, :, self.reconstruction_channel].unsqueeze(-1)
        return self.quantizer.quantize(x)

    def get_mean_field_preds(self, logits):
        """Return mean field approximation of the reconstruction for quick
        preview.
        """
        return self.quantizer.mean_field(logits)

    def loss(self, logits, targets):
        return self.quantizer.loss(logits, targets)

    def sample(self, x_priming, conds=(), start_step=0):
        with torch.no_grad():
            gen = x_priming.clone().detach()
            assert gen.size(3) == 1
            # For safety, put nans to make sure that we dont use data past
            # start_step to generate
            # gen[:, start_step:, :, :] = np.nan
            for r in range(start_step, gen.size(1)):
                for c in range(gen.size(2)):
                    logits = self(gen, conds)
                    gen[:, r, c, 0] = self.quantizer.sample(
                        logits[:, r, c, 0, :])
        return gen

    def plot_debug_samples(self, x_priming, conds=(), start_step=10):
        import matplotlib.pyplot as plt
        gen = self.sample(x_priming, conds, start_step)
        gen = gen.cpu()
        x_priming = x_priming.cpu()
        f = plt.Figure(figsize=(6, 1.5 * (gen.size(0))), dpi=300)
        for i in range(gen.size(0)):
            ax = f.add_subplot(gen.size(0), 2, 2 * i + 1)
            ax.imshow(x_priming[i, :, :, 0].transpose(0, 1))
            ax.axvline(start_step)
            ax.set_title('orig')
            ax = f.add_subplot(gen.size(0), 2, 2 * i + 2)
            ax.axvline(start_step)
            ax.set_title('gen')
            ax.imshow(gen[i, :, :, 0].transpose(0, 1))
        return f


class SimplePixelCNN(BasePixelCNN):
    def __init__(self, n_layers=4, kernel_size=3,
                 hid_channels=32,
                 activation=nn.ReLU, in_bias_shape=None,
                 cond_channels=None, **kwargs):
        del in_bias_shape  # unused
        assert not cond_channels  # simple pix cnn is unconditional
        in_channels = 1  # We only support single channel inputs
        super(SimplePixelCNN, self).__init__(**kwargs)
        self.layers = nn.Sequential()
        self.layers.add_module(
            'in_conv',
            MaskedConv2D(in_channels, hid_channels, kernel_size,
                         mask_self=True))
        self.layers.add_module('in_act', activation())
        for l in range(n_layers - 1):
            self.layers.add_module(
                f'hid_conv_{l}',
                MaskedConv2D(hid_channels, hid_channels, kernel_size,
                             mask_self=False))
            self.layers.add_module(f'hid_act_{l}', activation())
        self.layers.add_module(
            'out_conv',
            MaskedConv2D(hid_channels,
                         in_channels * self.quantizer.num_levels, kernel_size,
                         mask_self=False))

    def forward(self, input, conds=()):
        """
        input has shape N x W x H x 1
        output has shape N x out_channels x W x H
        """
        assert not conds
        x = input.permute(0, 3, 1, 2)
        x = self.layers(x)
        return x.permute(0, 2, 3, 1).unsqueeze(3)


class ResidualPixelCNN(BasePixelCNN):
    def __init__(self,
                 n_layers=7, out_n_layers=2, hid_channels=32,
                 in_kernel_size=7, kernel_size=3, in_bias_shape=1,
                 cond_channels=(), activation=torch.relu, **kwargs):
        in_channels = 1  # We only support single channel inputs
        super(ResidualPixelCNN, self).__init__(**kwargs)
        self.activation = activation
        in_bias_shape = _pair(in_bias_shape)
        self.bias = nn.Parameter(
            torch.zeros((1, 2 * hid_channels,) + in_bias_shape))
        nn.init.normal_(self.bias)
        self.in_to_res = MaskedConv2D(
            in_channels, 2 * hid_channels, in_kernel_size,
            mask_self=True, bias=False)
        self.res_to_hid = nn.ModuleList([
            nn.Conv2d(2 * hid_channels, hid_channels, 1)
            for _ in range(n_layers)])
        self.hid_to_hid = nn.ModuleList([
            MaskedConv2D(
                hid_channels, hid_channels, kernel_size, mask_self=False)
            for _ in range(n_layers)])
        self.cond_to_hid = nn.ModuleList([
            ConditioningAdder(cond_channels, hid_channels)
            for _ in range(n_layers)])
        self.hid_to_res = nn.ModuleList([
            nn.Conv2d(hid_channels, 2 * hid_channels, 1)
            for _ in range(n_layers)])

        self.res_to_out = nn.ModuleList([
            nn.Conv2d(2 * hid_channels, 2 * hid_channels, 1)
            for _ in range(out_n_layers - 1)] + [
            nn.Conv2d(2 * hid_channels, self.quantizer.num_levels, 1)])

    def forward(self, x, cond=()):
        """
        input has shape N x 1 x W x H
        output has shape N x out_channels x W x H
        """
        activation = self.activation
        res = self.in_to_res(x) + self.bias
        for res_to_hid, hid_to_hid, cond_to_hid, hid_to_res in zip(
                self.res_to_hid, self.hid_to_hid, self.cond_to_hid,
                self.hid_to_res):
            hid = res_to_hid(activation(res))
            hid = hid_to_hid(activation(hid))
            hid = cond_to_hid(hid, cond)
            res = res + hid_to_res(activation(hid))
        for res_to_out in self.res_to_out:
            res = res_to_out(activation(res))
        return res


class GatedPixelCNN(BasePixelCNN):
    def __init__(self,
                 n_layers=7, out_n_layers=2, hid_channels=32,
                 in_kernel_size=7, kernel_size=3, in_bias_shape=1,
                 dilation_base=(2, 1), len_dilation_stage=1,
                 cond_channels=(), **kwargs):
        in_channels = 1  # We only support single channel inputs
        super(GatedPixelCNN, self).__init__(**kwargs)
        in_bias_shape = _pair(in_bias_shape)
        dilation_base = _pair(dilation_base)
        self.bias = nn.Parameter(
            torch.zeros((1, hid_channels,) + in_bias_shape))
        nn.init.normal_(self.bias)
        self.in_to_res = MaskedConv2D(
            in_channels, hid_channels, in_kernel_size,
            mask_self=True, bias=False)
        self.gated_layers = nn.ModuleList([
            GatedMaskedStack2D(hid_channels, hid_channels, kernel_size,
                               bias_shape=in_bias_shape,
                               cond_channels=cond_channels,
                               mask_self='v')] + [
            GatedMaskedStack2D(hid_channels, hid_channels, kernel_size,
                               bias_shape=in_bias_shape,
                               cond_channels=cond_channels,
                               dilation=(
                                   dilation_base[0]**(l % len_dilation_stage),
                                   dilation_base[1]**(l % len_dilation_stage)),
                               mask_self=False)
            for l in range(n_layers - 1)])

        self.hid_to_res = nn.ModuleList([
            nn.Conv2d(hid_channels, hid_channels, 1)
            for _ in range(n_layers)])

        self.res_to_out = nn.ModuleList([
            nn.Conv2d(hid_channels, hid_channels, 1)
            for _ in range(out_n_layers - 1)] + [
            nn.Conv2d(hid_channels, self.quantizer.num_levels, 1)])

    def forward(self, x, cond=()):
        """
        input has shape N x W x H x 1
        output has shape N x out_channels x W x H
        """
        x = x.permute(0, 3, 1, 2)
        # v = x
        res = self.in_to_res(x) + self.bias
        h = res
        v = res
        for gated_layer, hid_to_res in zip(self.gated_layers, self.hid_to_res):
            v, h, skip = gated_layer(v, h, cond)
            res = res + hid_to_res(torch.relu(skip))
        for res_to_out in self.res_to_out:
            res = res_to_out(torch.relu(res))
        return res.permute(0, 2, 3, 1).unsqueeze(3)


class LookaheadGatedPixelCNN(GatedPixelCNN):
    """Predicts a pixel conditioned on past frames, skipping the most recent

    Args:
        ahead_frames: int, up to how many frames skip
        ahead_fraction: float, apply skipping to a random fraction of samples
        bidirectional: bool, condition also on the right context
    """
    def __init__(self, ahead_frames=2, ahead_fraction=0.5, bidirectional=False,
                 **kwargs):
        assert not (ahead_frames == 0 and ahead_fraction is not None)
        self.ahead_frames = ahead_frames
        self.ahead_fraction = ahead_fraction
        self.bidirectional = bidirectional
        super(LookaheadGatedPixelCNN, self).__init__(**kwargs)

    def forward(self, x, cond=()):
        """
        input has shape N x W x H x 1
        output has shape N x out_channels x W x H
        """
        x = x.permute(0, 3, 1, 2)

        if self.ahead_fraction is not None:
            probs = (np.ones((self.ahead_frames + 1,), dtype=np.float32) *
                     self.ahead_fraction / self.ahead_frames)
            probs[0] = 1.0 - self.ahead_fraction
            nframes = np.random.choice(self.ahead_frames + 1, p=probs)
        else:
            nframes = self.ahead_frames

        res_past_future = []
        contexts = ('past', 'future') if self.bidirectional else ('past',)
        for ctx in contexts:
            if nframes == 0:
                x_shift = x
            elif ctx == 'past':  # Apply padding on the time axis (dim=2)
                x_shift = F.pad(x, (0, 0, nframes, 0))[:, :, :-nframes, :]
            elif ctx == 'future':
                x_shift = F.pad(x, (0, 0, 0, nframes))[:, :, nframes:, :]

            if ctx == 'future':
                x_shift = torch.flip(x_shift, dims=[2, 3])

            # v = x
            res = self.in_to_res(x_shift) + self.bias
            h = res
            v = res
            for gated_layer, hid_to_res in zip(self.gated_layers,
                                               self.hid_to_res):
                v, h, skip = gated_layer(v, h, cond)
                res = res + hid_to_res(torch.relu(skip))
            res_past_future.append(res)

        res = torch.cat(res_past_future, dim=1)
        for res_to_out in self.res_to_out:
            res = res_to_out(torch.relu(res))
        return res.permute(0, 2, 3, 1).unsqueeze(3)
