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
from torch import distributions
from torch import nn
import torch.nn.functional as F

from distsup import utils


class ConvStack2D(nn.Module):
    """Stack of strided, dilated, and vanilla 2D convs.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hid_channels=64,
                 num_strided=0,  # stride Strided convs
                 strides=(2,),
                 kernel_size=3,
                 image_height=None,
                 ):
        del image_height  # unused
        super(ConvStack2D, self).__init__()
        self.layers = []
        self.length_reduction = 1
        for i in range(num_strided):
            stride = strides[min(i, len(strides) - 1)]
            self.length_reduction *= stride
            self.layers.append(nn.Conv2d(in_channels if i == 0 else hid_channels,
                                         hid_channels, padding=kernel_size // 2,
                                         kernel_size=kernel_size,
                                         stride=stride, bias=False))
            self.layers.append(nn.BatchNorm2d(hid_channels))
            self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(hid_channels, out_channels,
            padding=kernel_size // 2,
            kernel_size=kernel_size))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x, lens=None):
        x = x.permute(0, 3, 2, 1)
        x = self.layers(x)
        x = x.permute(0, 3, 2, 1)
        if lens is None:
            return x
        else:
            lens = (lens + self.length_reduction - 1) // self.length_reduction
            return x, lens


class ConvStack1D(nn.Module):
    """Stack of strided, dilated, and vanilla 1D convs.

    Input and output are N x W x H x C
    """
    def __init__(self,
                 in_channels,
                 image_height=1,
                 hid_channels=64,
                 num_preproc=0,  # 3 Convs with no stride no dilation
                 num_strided=0,  # 2 * stride Strided convs
                 strides=(2,),
                 kernels=None,
                 num_dilated=0,  # 3 Convs with exponentially growing rate
                 num_postdil=0,  # 3 Convs with no stride no dilation
                 num_postproc=0,  # 1 Convs to add more depth
                 ):
        super(ConvStack1D, self).__init__()
        self.in_to_hid = nn.Conv1d(in_channels * image_height, hid_channels, 1)
        self.preproc_layers = nn.ModuleList([
            nn.Conv1d(hid_channels, hid_channels, 3, padding=1)
            for _ in range(num_preproc)])
        strided = []
        self.length_reduction = 1
        for i in range(num_strided):
            stride = strides[min(i, len(strides) - 1)]
            self.length_reduction *= stride
            strided.append(nn.ReLU())
            strided.append(nn.ConstantPad1d((stride, stride - 1), 0))
            if kernels is not None and i < len(kernels):
                kernel = kernels[i]
            else:
                kernel = 2 * stride
            strided.append(
                nn.Conv1d(hid_channels, hid_channels, kernel, stride))
        self.strided_layers = nn.Sequential(*strided)
        self.dil_layers = nn.ModuleList([
            nn.Conv1d(hid_channels, hid_channels, 3,
                      dilation=2**i, padding=2**i)
            for i in range(num_dilated)
            ] + [
            nn.Conv1d(hid_channels, hid_channels, 3, padding=1)
            for _ in range(num_postdil)
            ] + [
            nn.Conv1d(hid_channels, hid_channels, 1)
            for _ in range(num_postproc)
            ])

    def forward(self, x, lens=None):
        N, W, H, C = x.size()
        x = x.view(N, W, H * C).permute(0, 2, 1)
        # x is N C W now
        x = self.in_to_hid(x)
        for l in self.preproc_layers:
            x = x + l(torch.relu(x))
        x = self.strided_layers(x)
        for l in self.dil_layers:
            x = x + l(torch.relu(x))
        x = torch.relu(x)
        x = x.permute(0, 2, 1).unsqueeze(2)
        if lens is None:
            return x
        else:
            lens = (lens + self.length_reduction - 1) // self.length_reduction
            return x, lens


class NDRLResidual(nn.Module):

    def __init__(self, hid=256, batch_norm=False, relu_before_add=True,
                 relu_first=False):
        super(NDRLResidual, self).__init__()
        if relu_first:
            assert relu_before_add
            layers = [
                nn.ReLU(),
                nn.Conv2d(hid, hid, kernel_size=3, padding=1),
                nn.BatchNorm2d(hid, affine=False) if batch_norm else None,
                nn.ReLU(),
                nn.Conv2d(hid, hid, kernel_size=1, padding=0),
                nn.BatchNorm2d(hid, affine=False) if batch_norm else None]
        else:
            layers = [
                nn.Conv2d(hid, hid, kernel_size=3, padding=1),
                nn.BatchNorm2d(hid, affine=False) if batch_norm else None,
                nn.ReLU(),
                nn.Conv2d(hid, hid, kernel_size=1, padding=0),
                nn.BatchNorm2d(hid, affine=False) if batch_norm else None,
                nn.ReLU() if relu_before_add else None]
        self.conv = nn.Sequential(*[l for l in layers if l is not None])
        self.relu_before_add = relu_before_add

    def forward(self, x):
        if self.relu_before_add:
            return x + self.conv(x)
        else:
            return F.relu(x + self.conv(x))


class NDRLEncoder(nn.Module):
    """Encoder for the CIFAR10 model in [1]

    [1] Oord et al., "Neural Discrete Representation Learning", 2017.

    """
    def __init__(self, in_channels, image_height,
                 hid=256, out_dim=None, num_layers=2, batch_norm=False,
                 out_relu=False):
        # Reduces 32x32 to 8x8
        super(NDRLEncoder, self).__init__()
        del in_channels  # unused
        del image_height  # unused
        assert num_layers >= 2
        # 32x32 is reduced to 8x8, but we make HW a single dimension 8x8=64
        # in order to embed every field of the map with VQBottleneck
        self.length_reduction = 0.5
        layers = [nn.Conv2d(3, hid, kernel_size=4, stride=2, padding=1),
                  nn.BatchNorm2d(hid, affine=False) if batch_norm else None,
                  nn.ReLU(),
                  nn.Conv2d(hid, hid, kernel_size=4, stride=2, padding=1),
                  nn.BatchNorm2d(hid, affine=False) if batch_norm else None,
                  nn.ReLU()]
        for i in range(2, num_layers):
            layers.extend([
                nn.Conv2d(hid, hid, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hid, affine=False) if batch_norm else None,
                nn.ReLU()])
        self.conv = nn.Sequential(*[l for l in layers if l is not None])
        self.residual = nn.Sequential(NDRLResidual(batch_norm=batch_norm),
                                      NDRLResidual(batch_norm=batch_norm))
        layers = [
            nn.Conv2d(hid, out_dim or hid, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim, affine=False) if batch_norm else None,
            nn.ReLU() if out_relu else None,
        ]
        self.out_conv = nn.Sequential(*[l for l in layers if l is not None])

    def forward(self, x, x_len=None):
        del x_len  # unused
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.residual(x)
        x = self.out_conv(x)
        # NCHW -> NHWC
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(x.size(0), x.size(1)*x.size(2), 1, -1)
        return x


class NDRLLucasEncoder(NDRLEncoder):

    def __init__(self, in_channels, image_height, hid=256, out_dim=None):
        super(NDRLLucasEncoder, self).__init__(in_channels=in_channels, image_height=image_height)
        # Reduces 32x32 to 8x8
        del in_channels  # unused
        del image_height  # unused

        # 32x32 is reduced to 16x16, but we make HW a single dimension
        # 16x16=256
        # in order to embed every field of the map with VQBottleneck
        self.length_reduction = 0.125
        layers = [nn.Conv2d(3, hid//2, kernel_size=4, stride=2, padding=1),
                  nn.ReLU(),
                  nn.Conv2d(hid//2, hid, kernel_size=3, stride=1, padding=1)]
        self.conv = nn.Sequential(*[l for l in layers if l is not None])
        self.residual = nn.Sequential(NDRLResidual(relu_first=True),
                                      NDRLResidual(relu_first=True))
        layers = [
            nn.Conv2d(hid, out_dim or hid, kernel_size=1, stride=1, padding=0),
        ]
        self.out_conv = nn.Sequential(*[l for l in layers if l is not None])


class NDRLReconstructor(nn.Module):
    """Decoder for the CIFAR10 model in [1]

    [1] Oord et al., "Neural Discrete Representation Learning", 2017.

    """
    def __init__(self, image_height, cond_channels=None,
                 hid=256, in_dim=None, num_layers=2, batch_norm=False,
                 output_type='continuous', fc_with_softmax=False,
                 output_std_bias_init=None):
        super(NDRLReconstructor, self).__init__()
        del image_height  # unused
        del cond_channels  # unused
        assert num_layers >= 2
        assert output_type in ('continuous', 'discrete', 'gaussian')
        self.output_type = output_type

        layers = [
            nn.ConvTranspose2d(in_dim or hid, hid, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hid, affine=False) if batch_norm else None,
            nn.ReLU()]
        self.in_conv = nn.Sequential(*[l for l in layers if l is not None])

        self.residual = nn.Sequential(NDRLResidual(batch_norm=batch_norm),
                                      NDRLResidual(batch_norm=batch_norm))
        layers = []
        for i in range(num_layers - 2):
            layers.extend([
                nn.ConvTranspose2d(hid, hid, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hid, affine=False) if batch_norm else None,
                nn.ReLU()])
        out_ch = 3 * 256 if output_type == 'discrete' else 3
        layers.extend([
            nn.ConvTranspose2d(hid, hid, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hid, affine=False) if batch_norm else None,
            nn.ReLU(),
            nn.ConvTranspose2d(hid, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch, affine=False) if batch_norm else None,
            nn.ReLU()])
        self.conv = nn.Sequential(*[l for l in layers if l is not None])

        if self.output_type == 'gaussian':
            sz = 32 * 32 * 3
            self.out_fc = nn.Linear(sz, sz * 2)
            if output_std_bias_init:
                self.out_fc.bias.data[sz:].fill_(output_std_bias_init)

    def get_inputs_and_targets(self, feats, feat_lens=None):
        del feat_lens  # unused
        return feats, feats

    def get_mean_field_preds(self, feats):
        return feats  # self.quantizer.mean_field(feats)

    def forward(self, inputs, conds):
        del inputs  # unused
        x = conds[0]
        hw = int(np.sqrt(x.size(1)))
        x = x.reshape(x.size(0), hw, hw, x.size(3))
        x = x.permute(0, 3, 1, 2)
        x = self.in_conv(x)
        x = self.residual(x)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        if self.output_type == 'gaussian':
            # NHWC --> N(HWC) --> NHWC2
            x = self.out_fc(x.contiguous().view(x.size(0), -1)).view(*x.size(), 2)
        return x

    def loss(self, logits, targets):
        if self.output_type == 'continuous':
            return F.mse_loss(logits, targets, reduction='mean')
        elif self.output_type == 'discrete':
            n, h, w, d = logits.size()
            return F.cross_entropy(
                logits.view(n, h, w, d // 256, 256).permute(0, 4, 1, 2, 3),
                ((targets / 2.0 + 0.5) * 255.0).long(),
                reduction='mean')
        elif self.output_type == 'gaussian':
            norm = distributions.normal.Normal(
                logits[:,:,:,:,0].contiguous().view(-1),
                torch.exp(logits[:,:,:,:,1].contiguous().view(-1)))
            return -torch.mean(norm.log_prob(targets.contiguous().view(-1)))
        else:
            raise ValueError


class NDRLLucasReconstructor(NDRLReconstructor):
    """Decoder for the CIFAR10 model in [1]

    [1] Oord et al., "Neural Discrete Representation Learning", 2017.

    """
    def __init__(self, image_height, cond_channels=None,
                 hid=256, in_dim=None):
        super(NDRLLucasReconstructor, self).__init__(image_height=image_height)
        del image_height  # unused
        del cond_channels  # unused

        layers = [
            nn.Conv2d(in_dim or hid, hid, kernel_size=3, stride=1, padding=1)]
        self.in_conv = nn.Sequential(*[l for l in layers if l is not None])

        self.residual = nn.Sequential(NDRLResidual(relu_first=True),
                                      NDRLResidual(relu_first=True),
                                      nn.ReLU())
        self.output_type = 'discrete'
        out_ch = 3 * 256
        layers = [
            nn.ConvTranspose2d(hid, out_ch, kernel_size=4, stride=2, padding=1)]
        self.conv = nn.Sequential(*[l for l in layers if l is not None])
