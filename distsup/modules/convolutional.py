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

import torch
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
