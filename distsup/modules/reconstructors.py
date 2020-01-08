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

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


import distsup.modules.pixcnn
import distsup.modules.wavenet
from distsup import utils
from distsup.modules import bert, convolutional, pixcnn, quantizers, wavenet
from distsup.utils import safe_squeeze
from distsup.modules.conditioning import CondNormNd, SpadeNd
from distsup.modules.misc import Exp


class BaseReconstructor(nn.Module):
    """Interface for documentation only
    """

    def __init__(self,
                 quantizer=dict(class_name=quantizers.L1Loss),
                 **kwargs):
        super(BaseReconstructor, self).__init__(**kwargs)
        self.quantizer = utils.construct_from_kwargs(quantizer)

    def get_inputs_and_targets(self, x, x_lens=None):
        """Quantize / transform inputs and produce reconstruction targets.

        Args:
            x: tensor of shape BS x T x H x C, normalized to 0-1 range.

        Returns
            inputs: tensor of shape BS x T x H x C
            targets: tensor of shape BS x T x H x C
        """
        del x_lens  # unused
        return self.quantizer.quantize(x)

    def get_mean_field_preds(self, logits):
        """Return mean field approximation of the rconstruction for quick
        preview.

        Args:
            logits: tensor of shape BS x T x H x C x Num_Levels

        Returns:
            tensor of shape BS x T x H x C
        """
        return self.quantizer.mean_field(logits)

    def loss(self, logits, targets):
        """ Compute apropriate loss, assuming a format of logits.
        """
        return self.quantizer.loss(logits, targets)

    def sample(self, x_priming, conds=(), start_step=0):
        """ """


class UpsamplingConv2d(nn.Module):
    def __init__(self,
            cond_channels=(),
            image_height=32,
            hid_channels=128,
            normalization=dict(class_name=nn.BatchNorm2d),
            quantizer=dict(class_name=quantizers.L1Loss)):
        super(UpsamplingConv2d, self).__init__()

        assert len(cond_channels) <= 2
        if len(cond_channels) == 1:
            assert normalization['class_name'] is torch.nn.BatchNorm2d
        else:
            assert normalization['class_name'] is not torch.nn.BatchNorm2d, (
                    "You can't use batchnorm with {} conditionings".format(
                        len(cond_channels)))
            normalization['cond_channels'] = cond_channels[1]['cond_dim']
        self.quantizer = utils.construct_from_kwargs(quantizer)

        scale = cond_channels[0]['reduction_factor']
        dim_vq = cond_channels[0]['cond_dim']

        self.conv = nn.Conv1d(dim_vq, image_height // scale * 64, kernel_size=1)
        num_strided = int(math.log2(scale))

        self.decoder = [
                nn.Conv2d(64, hid_channels, kernel_size=5, padding=2),
                utils.construct_from_kwargs(
                    {'num_features': hid_channels, **normalization}),
                nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(num_strided):
            self.decoder += [
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(hid_channels, hid_channels, kernel_size=5, padding=2),
                utils.construct_from_kwargs(
                    {'num_features': hid_channels, **normalization}),
                nn.ReLU(inplace=True)
            ]
        self.decoder.append(nn.Conv2d(hid_channels, self.quantizer.num_levels, 1))
        self.decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*self.decoder)

    def get_inputs_and_targets(self, feats, feats_lens=None):
        del feats_lens  # unused
        return self.quantizer.quantize(feats)

    def get_mean_field_preds(self, feats):
        return self.quantizer.mean_field(feats)

    def loss(self, logits, targets):
        recon = self.quantizer.loss(logits, targets)
        return recon

    def plot_debug_samples(self, x_priming, conds=(), start_step=10):
        import matplotlib.pyplot as plt
        gen = self(x_priming, conds)
        gen = gen.cpu()
        x_priming = x_priming.cpu()
        f = plt.Figure(figsize=(6, 1.5 * (gen.size(0))), dpi=300)
        for i in range(gen.size(0)):
            ax = f.add_subplot(gen.size(0), 2, 2 * i + 1)
            ax.imshow(x_priming[i, 0].transpose(0, 1))
            ax.axvline(start_step)
            ax.set_title('orig')
            ax = f.add_subplot(gen.size(0), 2, 2 * i + 2)
            ax.axvline(start_step)
            ax.set_title('gen')
            ax.imshow(gen[i, 0].transpose(0, 1))
        return f

    def decode(self, z, cond=None):
        for m in self.decoder:
            if isinstance(m, (CondNormNd, SpadeNd)):
                z = m(z, cond)
            else:
                z = m(z)
        return z

    def forward(self, input, conds):
        del input  # unused
        #z = conds[0].permute(0, 3, 2, 1)
        z = self.conv(conds[0].view(conds[0].size(0), conds[0].size(1), -1).permute(0, 2, 1))
        z = z.view(z.size(0), 64, -1, z.size(2)).permute(0, 1, 2, 3)
        cond = None
        if len(conds) > 1:
            cond = conds[1]
            # Squeeze out the width and heigth,
            # as we assume this conditionig is global
            cond = safe_squeeze(cond, 1)
            cond = safe_squeeze(cond, 1)
        x2 = self.decode(z, cond=cond)

        return x2.permute(0, 3, 2, 1).unsqueeze(3)


class NullReconstructor(nn.Module):
    def __init__(self, **kwargs):
        del kwargs  # unused
        super(NullReconstructor, self).__init__()

    def get_inputs_and_targets(self, x, x_lens=None):
        del x_lens  # unused
        return x, x

    def forward(self, x, *args, **kwargs):
        return x

    def get_mean_field_preds(self, logits):
        return None

    def loss(self, logits, targets):
        return torch.zeros_like(logits)

    def plot_debug_samples(self, *args, **kwargs):
        return None


class ColumnGatedPixelCNN(pixcnn.GatedPixelCNN):
    """
    A pixel CNN that is generates the data form left to right and row 0 upward.

    The internal assumed image format is BS x C x T x H
    """

    def __init__(self, image_height, **kwargs):
        super(ColumnGatedPixelCNN, self).__init__(
            in_bias_shape=(1, image_height),
            **kwargs)


class LookaheadColumnGatedPixelCNN(pixcnn.LookaheadGatedPixelCNN):
    """
    A pixel CNN that is generates the data form left to right and row 0 upward.

    The internal assumed image format is BS x C x T x H
    """

    def __init__(self, image_height, **kwargs):
        super(LookaheadColumnGatedPixelCNN, self).__init__(
              in_bias_shape=(1, image_height),
              **kwargs)


class ColumnwiseWaveNet(nn.Module):
    """A WaveNet that predicts whole columns at once.

    The internal image format is BS x C x T with C assumed to be the height.
    """
    def __init__(self,
                 image_height,
                 reconstruction_channel=None,
                 wave_net=dict(
                    class_name=wavenet.WaveNet,
                 ),
                 quantizer=dict(class_name=quantizers.BinaryXEntropy),
                 **kwargs):
        super(ColumnwiseWaveNet, self).__init__()
        self.quantizer = utils.construct_from_kwargs(quantizer)
        if 'in_channels' in wave_net or 'out_channels' in wave_net:
            raise ValueError('Channels for ColumnwiseWaveNet are auto set')
        wave_net['in_channels'] = image_height
        wave_net['out_channels'] = image_height * self.quantizer.num_levels
        wave_net.update(kwargs)
        self.wave_net = utils.construct_from_kwargs(wave_net)
        self.reconstruction_channel = reconstruction_channel

    def get_inputs_and_targets(self, x, x_lens=None):
        """Premute the inputs to

        x has shape BS x T x H x 1
        """
        del x_lens  # unused
        if self.reconstruction_channel is None:
            assert x.size(3) == 1
        else:
            x = x[:, :, :, self.reconstruction_channel].unsqueeze(-1)
        return self.quantizer.quantize(x)

    def forward(self, x, conds=()):
        """
        x: BS x T x H x 1
        conds: list of BS x DimC x T/k
        """
        # make the height the channel
        x = safe_squeeze(x, 3).transpose(1, 2)
        logits = self.wave_net.forward(x, conds)
        # move the channel back to height
        logits = logits.transpose(1, 2)
        # add the channel dim, the logit
        logits = logits.reshape(
            [logits.size(0), logits.size(1), -1, 1, self.quantizer.num_levels])
        return logits

    def get_mean_field_preds(self, logits):
        return self.quantizer.mean_field(logits)

    def loss(self, logits, targets):
        return self.quantizer.loss(logits, targets)

    def sample(self, x_priming, conds=(), start_step=0):
        with torch.no_grad():
            gen = x_priming.clone().detach()
            # For safety, put nans to make sure that we dont use data past
            # start_step to generate
            gen[:, start_step:] = np.nan
            for col in range(start_step, gen.size(1)):
                logits = self(gen, conds)
                gen[:, col] = self.quantizer.sample(logits[:, col])
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


class LookaheadColumnwiseWaveNet(ColumnwiseWaveNet):
    def __init__(self, image_height, ahead_frames=2, ahead_corruption=None,
                 ahead_fraction=None, bidirectional=False, **kwargs):
        if 'wave_net' not in kwargs:
            kwargs['wave_net'] = dict(class_name=wavenet.LookaheadWaveNet)
        kwargs['wave_net'].update(dict(ahead_frames=ahead_frames,
                                       ahead_corruption=ahead_corruption,
                                       ahead_fraction=ahead_fraction,
                                       bidirectional=bidirectional))
        super(LookaheadColumnwiseWaveNet, self).__init__(
            image_height=image_height, **kwargs)


def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

def noop(x): return x

def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm2d(nf)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    layers = [conv(ni, nf, ks, stride=stride), bn]
    if act: layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, expansion, ni, nh, stride=1):
        super(ResBlock, self).__init__()
        nf,ni = nh*expansion,ni*expansion
        layers  = [conv_layer(ni, nh, 3, stride=stride),
                   conv_layer(nh, nf, 3, zero_bn=True, act=False)
        ] if expansion == 1 else [
                   conv_layer(ni, nh, 1),
                   conv_layer(nh, nh, 3, stride=stride),
                   conv_layer(nh, nf, 1, zero_bn=True, act=False)
        ]
        self.convs = nn.Sequential(*layers)
        # TODO: check whether act=True works better
        self.idconv = noop if ni==nf else conv_layer(ni, nf, 1, act=False)
        self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x): return F.relu(self.convs(x) + self.idconv(self.pool(x)), inplace=True)


class DownsamplingResBlock(nn.Module):
    def __init__(self, ni, nh, nf, stride=1):
        super(DownsamplingResBlock, self).__init__()
        self.convs = nn.Sequential(*[
            conv_layer(ni, nh, 1),
            conv_layer(nh, nh, 3, stride=stride),
            conv_layer(nh, nf, 1, zero_bn=True, act=False)])
        # TODO: check whether act=True works better
        if ni == nf and (type(stride) is int and stride == 1):
            self.idconv = noop
        else:
            self.idconv = conv_layer(ni, nf, stride=stride, act=False)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(x), inplace=True)


class Decoder_2d(nn.Module):
    """
    An image 2D decoder model with a quantized bottleneck.
    Args:
        encoder: ConvStack
    """

    def __init__(self,
                 image_height,
                 cond_channels=(),
                 stride=2,
                 hid_channels=256,
                 use_sigmoid=True,
                 use_pixelshuffle=True,
                 resblocks=2,
                 out_channels=1,
                 quantizer=dict(class_name=quantizers.L1Loss),
                 normalization=dict(class_name=torch.nn.BatchNorm2d),
                 **kwargs):
        super(Decoder_2d, self).__init__(**kwargs)
        self.quantizer = utils.construct_from_kwargs(quantizer)

        assert len(cond_channels) <= 2
        if len(cond_channels) == 1:
            assert normalization['class_name'] is torch.nn.BatchNorm2d
        else:
            assert normalization['class_name'] is not torch.nn.BatchNorm2d
            normalization['cond_channels'] = cond_channels[1]['cond_dim']

        scale = cond_channels[0]['reduction_factor']
        dim_vq = cond_channels[0]['cond_dim']

        self.conv = nn.Conv1d(dim_vq, (image_height + scale - 1) // scale * 64, kernel_size=1)

        num_strided = int(math.log(scale) / math.log(stride))
        padding, output_padding = [], []
        if stride == 3:
            for i in range(num_strided):
                padding.append(0)
                output_padding.append(0)
        elif stride == 2:
            for i in range(num_strided):
                if i % 2 == 0:
                    padding.append(1)
                    output_padding.append(0)
                else:
                    padding.append(0)
                    output_padding.append(1)
            if num_strided % 2 == 1:
                output_padding[-1] = 1
        if num_strided == 0:
            decoder = [
                dict(in_channels=64, out_channels=out_channels * self.quantizer.num_levels,
                     kernel_size=1, stride=1, padding=0, bias=True,
                     output_padding=0)
            ]
        else:
            decoder = [
                dict(in_channels=64, out_channels=hid_channels, padding=2, kernel_size=5, stride=1,
                     bias=False),
                *[dict(in_channels=hid_channels,
                       out_channels=hid_channels if idx != num_strided - 1 else out_channels * self.quantizer.num_levels,
                       kernel_size=3,
                       padding=padding[idx],
                       output_padding=output_padding[idx],
                       stride=stride,
                       bias=False if idx != num_strided - 1 else True) for idx in range(num_strided)],
            ]

        self.decoder = []
        for conv in decoder[:-1]:
            #if len(self.decoder) > 0:
            #    self.decoder.extend([ResBlock(1, hid_channels, hid_channels)] * resblocks)
            if (not use_pixelshuffle):
                self.decoder += [
                    nn.ConvTranspose2d(**conv),
                    utils.construct_from_kwargs(
                        {'num_features': conv['out_channels'], **normalization}),
                    nn.ReLU(inplace=True)
                ]
            else:
                if conv['kernel_size'] == 5:
                    self.decoder += [
                        ResBlock(stride * stride, 64 // (stride * stride), hid_channels // (stride * stride)),
                        utils.construct_from_kwargs(
                            {'num_features': conv['out_channels'], **normalization}),
                        nn.ReLU(inplace=True)
                    ]
                else:
                    self.decoder += [
                        #nn.Conv2d(hid_channels, hid_channels * (stride * stride), kernel_size=3, padding=1),
                        ResBlock(stride * stride, hid_channels // (stride * stride), hid_channels),
                        nn.PixelShuffle(stride),
                        utils.construct_from_kwargs(
                            {'num_features': conv['out_channels'], **normalization}),
                        nn.ReLU(inplace=True)
                    ]
        if use_pixelshuffle:
            self.decoder.append(nn.Conv2d(hid_channels, out_channels * self.quantizer.num_levels * (stride * stride), kernel_size=3, padding=1))
            self.decoder.append(nn.PixelShuffle(stride))
        else:
            self.decoder.append(nn.ConvTranspose2d(**decoder[-1]))
        if use_sigmoid:
            self.decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*self.decoder)

        self.apply(utils.conv_weights_xavier_init)

    def encode(self, x):
        return self.conv(x)

    def decode(self, z, cond=None):
        for m in self.decoder:
            if isinstance(m, (CondNormNd, SpadeNd)):
                z = m(z, cond)
            else:
                z = m(z)
        return z

    def forward(self, input, conds):
        z = self.encode(conds[0].view(conds[0].size(0), conds[0].size(1), -1).permute(0, 2, 1))
        z = z.view(z.size(0), 64, -1, z.size(2)).permute(0, 1, 3, 2)
        cond = None
        if len(conds) > 1:
            cond = conds[1]
            # Squeeze out the width and heigth,
            # as we assume this conditionig is global
            cond = safe_squeeze(cond, 1)
            cond = safe_squeeze(cond, 1)
        x2 = self.decode(z, cond=cond)
        x2 = x2.view(x2.size(0), -1, self.quantizer.num_levels, x2.size(2), x2.size(3)).permute(0, 3, 4, 1, 2)

        # Truncate x2 to input size
        _, _, h_in, *_ = input.shape
        _, _, h_x2, *_ = x2.shape

        assert h_x2 >= h_in, f"The reconstruction ({x2.shape}) must be as large as the input {input.shape}."

        h_d = h_x2 - h_in
        h_s = h_d // 2
        h_e = h_s + h_in

        x2 = x2[:, :, h_s:h_e, :]

        return x2

    def get_inputs_and_targets(self, feats, feats_lens=None):
        del feats_lens  # unused
        return self.quantizer.quantize(feats)

    def get_mean_field_preds(self, feats):
        return self.quantizer.mean_field(feats)

    def loss(self, logits, targets):
        recon = self.quantizer.loss(logits, targets)
        return recon

    def plot_debug_samples(self, x_priming, conds=(), start_step=10):
        import matplotlib.pyplot as plt
        gen = self(x_priming, conds)
        gen = gen.cpu()
        x_priming = x_priming.cpu()
        f = plt.Figure(figsize=(6, 1.5 * (gen.size(0))), dpi=300)
        for i in range(gen.size(0)):
            ax = f.add_subplot(gen.size(0), 2, 2 * i + 1)
            ax.imshow(x_priming[i, 0].transpose(0, 1))
            ax.axvline(start_step)
            ax.set_title('orig')
            ax = f.add_subplot(gen.size(0), 2, 2 * i + 2)
            ax.axvline(start_step)
            ax.set_title('gen')
            ax.imshow(gen[i, 0].transpose(0, 1))
        return f


class DownsamplingDecoder2D(nn.Module):
    """
    """
    def __init__(self,
                 input_dim,
                 image_height,
                 num_layers,
                 in_channels=64,
                 out_channels=1,
                 hid_channels=64,
                 use_sigmoid=True,
                 quantizer=dict(class_name=quantizers.L1Loss),
                 normalization=dict(class_name=torch.nn.BatchNorm2d),
                 **kwargs):
        super(DownsamplingDecoder2D, self).__init__(**kwargs)
        self.quantizer = utils.construct_from_kwargs(quantizer)

        self.image_height = image_height
        self.hid_channels = hid_channels
        self.conv_input_dim = input_dim // image_height
        self.conv = nn.Conv1d(input_dim, 64 * image_height, kernel_size=1)

        conv_stack = []
        for _ in range(num_layers - 1):
            conv_stack += [
                DownsamplingResBlock(hid_channels, hid_channels, hid_channels),
                utils.construct_from_kwargs(
                    {'num_features': hid_channels, **normalization}),
                nn.ReLU(inplace=True)]

        conv_stack += [
            nn.Conv2d(hid_channels, out_channels * self.quantizer.num_levels,
                      stride=1, kernel_size=3, padding=1)]
        if use_sigmoid:
            conv_stack += [nn.Sigmoid()]

        self.conv_stack = nn.Sequential(*conv_stack)
        self.apply(utils.conv_weights_xavier_init)

    def forward(self, x):
        # x: (bsz x dim x t)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), self.hid_channels, self.image_height, x.size(-1))
        x2 = self.conv_stack(x)
        x2 = x2.view(x2.size(0), -1, self.quantizer.num_levels,
                     x2.size(2), x2.size(3))
        # (bs x 1 x 1 x h x t) -> (bs x t x 1 x h x 1)
        # XXX This should squeeze out 1 and leave channels as '3', but not sure
        return utils.safe_squeeze(x2.permute(0, 4, 3, 1, 2), -1)

    def get_inputs_and_targets(self, feats):
        return self.quantizer.quantize(feats)

    def get_mean_field_preds(self, feats):
        return self.quantizer.mean_field(feats)

    def loss(self, logits, targets):
        recon = self.quantizer.loss(logits, targets)
        return recon

    def plot_debug_samples(self, x_priming, conds=(), start_step=10):
        import matplotlib.pyplot as plt
        gen = self(x_priming, conds)
        gen = gen.cpu()
        x_priming = x_priming.cpu()
        f = plt.Figure(figsize=(6, 1.5 * (gen.size(0))), dpi=300)
        for i in range(gen.size(0)):
            ax = f.add_subplot(gen.size(0), 2, 2 * i + 1)
            ax.imshow(x_priming[i, 0].transpose(0, 1))
            ax.axvline(start_step)
            ax.set_title('orig')
            ax = f.add_subplot(gen.size(0), 2, 2 * i + 2)
            ax.axvline(start_step)
            ax.set_title('gen')
            ax.imshow(gen[i, 0].transpose(0, 1))
        return f


class MLPSkip(nn.Module):
    def __init__(self, in_channels, image_height, hidden_size, num_layers, output=""):
        super(MLPSkip, self).__init__()
        for i in range(num_layers):
            if i == 0:
                i_size = in_channels
            else:
                i_size = hidden_size
            setattr(self, "lin%d" % i, nn.Linear(i_size, hidden_size))
            if i > 0:
                setattr(self, "skip%d" % i, nn.Linear(in_channels, hidden_size))

        if output == "gauss":
            self.mu = nn.Linear(hidden_size, image_height)
            self.scale = nn.Sequential(nn.Linear(hidden_size, image_height),
                                       Exp())
        else:
            self.out = nn.Linear(hidden_size, image_height)

        self.num_layers = num_layers
        self.output_type = output

    def forward(self, z):
        # bsz, w, h, c = z.size()
        # flatten the last two dimensions
        # z = z.contiguous().view(bsz, w, h*c)
        tmp = z
        z = F.relu(getattr(self, "lin0")(z))
        for i in range(1, self.num_layers):
            z = F.relu(getattr(self, "lin%d" % i)(z) + getattr(self, "skip%d" % i)(tmp))
        if self.output_type == "gauss":
            return self.mu(z), self.scale(z)
        else:
            return self.out(z)


class StackReconstructor(BaseReconstructor):
    def __init__(self, image_height, cond_channels,
                 stack=dict(class_name=convolutional.ConvStack1D, num_postproc=2),
                 out_channels=1, reconstruction_channel=None,
                 **kwargs):
        """
        Args:
            image_heigth: image_heigth of the reconstruction
            cond_channels: simensionality of conditioning
            stack: the rec stack to use
            out_channels: number of output channels
            reconstruction_channel: limit the reconsturction to only this
                                    channel. Forces out_channels = 1
        """
        super(StackReconstructor, self).__init__(**kwargs)
        self.reconstruction_channel = reconstruction_channel
        if self.reconstruction_channel is not None:
            assert out_channels == 1
        in_channels = sum([c['cond_dim'] for c in cond_channels])
        stack['in_channels'] = in_channels
        self.stack = utils.construct_from_kwargs(stack)
        self.stack.eval()
        stack_out_shape = self.stack(
            torch.empty((1, 100, 1, in_channels))).size()
        self.proj = nn.Linear(
            stack_out_shape[-1],
            image_height * out_channels * self.quantizer.num_levels)

    def get_inputs_and_targets(self, x, x_lens=None):
        """Premute the inputs to

        x has shape BS x T x H x 1
        """
        del x_lens  # unused
        if self.reconstruction_channel is not None:
            x = x[:, :, :, self.reconstruction_channel].unsqueeze(-1)
        return self.quantizer.quantize(x)

    def forward(self, x, conds):
        x_len = x.size(1)
        c_inputs = []
        for c in conds:
            c_len = c.size(1)
            assert (x_len % c_len) == 0
            c = c.repeat_interleave(x_len // c_len, 1)
            c_inputs.append(c)
        c_inputs = torch.cat(c_inputs, dim=3)
        out = self.stack(c_inputs)
        out = self.proj(out)
        out = out.view(x.size() + (-1,))
        return out


class RightToLeftReconstructor(nn.Module):
    """Interface for documentation only
    """

    def __init__(self,
                 wrapped_class_name,
                 **kwargs):
        super(RightToLeftReconstructor, self).__init__()
        self.reconstructor = utils.construct_from_kwargs(
            dict(class_name=wrapped_class_name), additional_parameters=kwargs)

    def get_inputs_and_targets(self, x, x_lens=None):
        del x_lens  # unused
        x = torch.flip(x, dims=(1,))
        # if x_lens is None:

        # else:
        #     x = utils.masked_flip(x, x_lens)
        return self.reconstructor.get_inputs_and_targets(x)

    def forward(self, x, conds=()):
        conds = [torch.flip(c, dims=[1]) for c in conds]
        return self.reconstructor(x, conds)

    def get_mean_field_preds(self, logits):
        preds = self.reconstructor.get_mean_field_preds(logits)
        return torch.flip(preds, dims=[1])

    def loss(self, logits, targets):
        losses = self.reconstructor.loss(logits, targets)
        return torch.flip(losses, dims=[1])
