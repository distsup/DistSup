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

import torch
import torch.nn as nn
import torch.nn.functional as F

_norms = [None,
    {
        'batch': nn.BatchNorm1d,
        'instance': nn.InstanceNorm1d,
        'layer': nn.LayerNorm,
    },
    {
        'batch': nn.BatchNorm2d,
        'instance': nn.InstanceNorm2d,
        'layer': nn.LayerNorm,
    },
    {
        'batch': nn.BatchNorm3d,
        'instance': nn.InstanceNorm3d,
        'layer': nn.LayerNorm,
    },
]

_convs = [None, nn.Conv1d, nn.Conv2d, nn.Conv3d]

class CondNormNd(nn.Module):
    def __init__(self, n_dim, num_features, cond_channels, norm_type='batch'):
        super(CondNormNd, self).__init__()

        self.norm = _norms[n_dim][norm_type](num_features, affine=False)

        self.make_bias = nn.Linear(cond_channels, num_features)
        self.make_weight = nn.Linear(cond_channels, num_features)
        self.n_dim = n_dim

    def forward(self, x, cond):
        size = x.shape[:2] + (1,) * self.n_dim

        weight = self.make_weight(cond).view(*size)
        bias = self.make_bias(cond).view(*size)
        x = self.norm(x)

        return (1 + weight) * x + bias

class CondNorm1d(CondNormNd):
    def __init__(self, num_features, cond_channels, norm_type='batch'):
        super(CondNorm1d, self).__init__(1, num_features, cond_channels, norm_type)

class CondNorm2d(CondNormNd):
    def __init__(self, num_features, cond_channels, norm_type='batch'):
        super(CondNorm2d, self).__init__(2, num_features, cond_channels, norm_type)

class CondNorm3d(CondNormNd):
    def __init__(self, num_features, cond_channels, norm_type='batch'):
        super(CondNorm3d, self).__init__(3, num_features, cond_channels, norm_type)


class SpadeNd(nn.Module):
    def __init__(self, n_dim, num_features, cond_channels, cond_hidden, cond_ks, norm_type='batch'):
        super(SpadeNd, self).__init__()

        self.norm = _norms[n_dim][norm_type](num_features, affine=False)

        self.initial = _convs[n_dim](cond_channels, cond_hidden, cond_ks,
                padding=cond_ks//2)
        self.make_weight = _convs[n_dim](cond_hidden, nh_channels, cond_ks,
                padding=cond_ks//2)
        self.make_bias = _convs[n_dim](cond_hidden, nh_channels, cond_ks,
                padding=cond_ks//2)

    def forward(self, x, cond):
        x = self.norm(x)

        cond = F.interpolate(cond, size=x.shape[2:], mode='nearest')
        cond_hidden = F.relu(self.initial(cond), inplace=True)
        weight = self.make_weight(cond_hidden)
        bias = self.make_bias(cond_hidden)

        return (1 + weight) * x + bias

class Spade1d(nn.Module):
    def __init__(self, **kwargs):
        super(Spade1d, self).__init__(n_dim=1, **kwargs)

class Spade2d(nn.Module):
    def __init__(self, **kwargs):
        super(Spade2d, self).__init__(n_dim=2, **kwargs)

class Spade3d(nn.Module):
    def __init__(self, **kwargs):
        super(Spade3d, self).__init__(n_dim=3, **kwargs)
