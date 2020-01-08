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

class Jitter(nn.Module):
    def __init__(self, model, prob=0.12, **kwargs):
        super(Jitter, self).__init__()
        self.model = utils.construct_from_kwargs(model, additional_parameters=kwargs)
        self.prob = prob

    def forward(self, x):
        if self.training:
            N, W, H, C = x.size()
            index = torch.arange(0, W).to(x)
            change = torch.bernoulli(index * 0 + self.prob * 2) # whether to change
            shift = torch.bernoulli(index * 0 + 0.5) * 2 - 1 # left or right
            index = index + change * shift
            index = index.long().clamp(0, W - 1)
            x = torch.index_select(x, dim=1, index=index)
        return self.model(x)
