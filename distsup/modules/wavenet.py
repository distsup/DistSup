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

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import distsup.utils as utils
from .pixcnn import ConditioningAdder


class CausalConv1D(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1, bias=True, mask_self=True):
        self.mask_self = mask_self
        # In 1D, self mask means more padding
        self.pad = dilation * (kernel_size - 1) + mask_self
        super(CausalConv1D, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation,
            groups, bias, 'zeros')

    def forward(self, x):
        """x is N x C x T"""
        x = F.pad(x, (self.pad, 0))
        ret = F.conv1d(x, self.weight, self.bias, self.stride,
                       (0,), self.dilation, self.groups)
        if self.mask_self:
            ret = ret[:, :, :-1]
        return ret


class GatedCConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 cond_channels=(), stride=1, dilation=1, groups=1,
                 bias=True, mask_self=True, act='tanh'):
        super(GatedCConv1D, self).__init__()
        self.out_chanels = out_channels
        self.conditioning = ConditioningAdder(cond_channels, 2*out_channels)
        self.conv = CausalConv1D(
            in_channels, out_channels * 2, kernel_size, stride,
            dilation, groups, bias, mask_self)
        self.act = getattr(torch, act)

    def forward(self, x, conds):
        """
        x: N x C x T
        conds: list of N x CC x T/k
        """
        gates = self.conv(x)
        gates = self.conditioning(gates, conds)
        g1, g2 = gates.chunk(2, dim=1)
        return torch.sigmoid(g1) * self.act(g2)


class WaveNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_channels=(),
                 hid_channels=32,
                 res_channels=128,
                 skip_channels=128,
                 num_layers=10,
                 num_output_layers=2,
                 num_stages=3,
                 kernel_size=2,
                 in_kernel_size=4,
                 dropout=0
                 ):
        super(WaveNet, self).__init__()
        self.dropout = dropout
        self.x_to_res = CausalConv1D(in_channels, res_channels,
                                     in_kernel_size, mask_self=True)
        self.res_to_skip = nn.Conv1d(res_channels, skip_channels, 1)
        self.res_to_hid = nn.ModuleList([
            GatedCConv1D(res_channels, hid_channels, kernel_size,
                         cond_channels, dilation=2**(i % num_stages),
                         mask_self=False)
            for i in range(num_layers)
            ])
        self.hid_to_skip = nn.ModuleList([
            nn.Conv1d(hid_channels, skip_channels, 1, bias=False)
            for _ in range(num_layers)
            ])
        self.hid_to_res = nn.ModuleList([
            nn.Conv1d(hid_channels, res_channels, 1, bias=False)
            for _ in range(num_layers - 1)
            ] + [None])
        self.skip_to_out = nn.ModuleList([
            nn.Conv1d(skip_channels, skip_channels, 1)
            for _ in range(num_output_layers - 1)] +
            [nn.Conv1d(skip_channels, out_channels, 1)])

    def forward(self, x, conds=()):
        """
        x: BS x Dim x T
        conds: list of BS x DimC x T/k
        """
        x_res = self.x_to_res(x)
        x_skip = self.res_to_skip(x_res)
        for res_to_hid, hid_to_skip, hid_to_res in zip(
                self.res_to_hid, self.hid_to_skip, self.hid_to_res):
            x_hid = res_to_hid(x_res, conds)
            x_hid = F.dropout(x_hid, self.dropout, self.training, True)
            x_skip += hid_to_skip(x_hid)
            if hid_to_res is None:
                x_res = None  # We don't use the last residual output
            else:
                x_res = x_res + hid_to_res(x_hid)
        for skip_to_out in self.skip_to_out:
            x_skip = torch.relu(x_skip)
            x_skip = F.dropout(x_skip, self.dropout, self.training, True)
            x_skip = skip_to_out(x_skip)
        return x_skip


class LookaheadWaveNet(WaveNet):
    """Predicts a pixel conditioned on past frames, skipping the most recent

    Args:
        ahead_frames: int, up to how many frames skip
        ahead_corruption: float, use skipped fragment, but randomly corrupted
        ahead_fraction: float, apply skipping to a random fraction of samples
        bidirectional: bool, condition also on the right context
    """
    def __init__(self, in_channels, out_channels, ahead_frames=2,
                 ahead_corruption=0.95, ahead_fraction=0.5,
                 bidirectional=False, **kwargs):
        assert not (ahead_frames == 0 and ahead_fraction is not None)
        self.ahead_frames = ahead_frames
        self.ahead_corruption = ahead_corruption
        self.ahead_fraction = ahead_fraction
        self.bidirectional = bidirectional

        if ahead_corruption is not None:
            in_channels *= 2
        super(LookaheadWaveNet, self).__init__(
            in_channels=in_channels, out_channels=out_channels, **kwargs)

    def forward(self, x, conds=()):
        """
        x: BS x Dim x T
        conds: list of BS x DimC x T/k
        """
        x_skip = 0

        if self.ahead_corruption is not None:
            ber = torch.distributions.bernoulli.Bernoulli(
                torch.tensor([1.0 - self.ahead_corruption], device=x.device))
            mask = utils.safe_squeeze(ber.sample(sample_shape=x.size()), -1)
            x_corrupt = x * mask

        if self.ahead_fraction is not None:
            probs = (np.ones((self.ahead_frames + 1,), dtype=np.float32) *
                     self.ahead_fraction / self.ahead_frames)
            probs[0] = 1.0 - self.ahead_fraction
            nframes = np.random.choice(self.ahead_frames + 1, p=probs)
        else:
            nframes = self.ahead_frames

        contexts = ('past', 'future') if self.bidirectional else ('past',)
        for ctx in contexts:
            if nframes == 0:
                x_shift = x
            elif ctx == 'past':  # Apply padding on the time axis (dim=2)
                x_shift = F.pad(x, (nframes, 0))[:, :, :-nframes]
            elif ctx == 'future':
                x_shift = F.pad(x, (0, nframes))[:, :, nframes:]

            if self.ahead_corruption is not None:  # Stack on the dim axis
                x_shift = torch.cat([x_shift, x_corrupt], dim=1)

            if ctx == 'future':
                x_shift = torch.flip(x_shift, dims=[2])

            x_res = self.x_to_res(x_shift)
            x_skip += self.res_to_skip(x_res)
            for res_to_hid, hid_to_skip, hid_to_res in zip(
                    self.res_to_hid, self.hid_to_skip, self.hid_to_res):
                x_hid = res_to_hid(x_res, conds)
                x_hid = F.dropout(x_hid, self.dropout, self.training, True)
                x_skip += hid_to_skip(x_hid)
                if hid_to_res is None:
                    x_res = None  # We don't use the last residual output
                else:
                    x_res = x_res + hid_to_res(x_hid)

        for skip_to_out in self.skip_to_out:
            x_skip = torch.relu(x_skip)
            x_skip = F.dropout(x_skip, self.dropout, self.training, True)
            x_skip = skip_to_out(x_skip)
        return x_skip
