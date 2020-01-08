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
from torch.distributions.laplace import Laplace
from torch.distributions.normal import Normal
import torch.nn
import torch.nn.functional as F
import torchvision.models as tvmodels

from distsup.utils import safe_squeeze


# Quantizations with losses
class BaseDataQuantizer(torch.nn.Module):
    def __init__(self, **kwargs):
        super(BaseDataQuantizer, self).__init__(**kwargs)
        self.num_levels = 1

    def quantize(self, x):
        return self(x)

    def forward(self, x):
        """Encode the values, e.g. by quantizeing.

        Args:
            x: the data to quantize
        Returns:
            tuple of:
                - tensor with encoded values
                - tensor with
        """
        return x, x

    def dequantize(self, x):
        return x

    def mean_field(self, logits):
        raise NotImplementedError

    def sample(self, logits):
        raise NotImplementedError

    def loss(self, x, targets):
        raise NotImplementedError


class SoftmaxUniformQuantizer(BaseDataQuantizer):
    def __init__(self, num_levels, min=0.0, max=1.0, **kwargs):
        assert min == 0.0, "Not implemented"
        assert max == 1.0, "Not implemented"
        super(SoftmaxUniformQuantizer, self).__init__(**kwargs)
        self.num_levels = num_levels

    def forward(self, x):
        assert x.min() >= 0.0
        assert x.max() <= 1.0
        targets = (x * self.num_levels).clamp(0, self.num_levels - 1).long()
        assert targets.min() >= 0
        assert targets.max() < self.num_levels
        inputs = self.dequantize(targets)
        return inputs, targets

    def dequantize(self, q):
        return q.float() / (self.num_levels - 1)

    def mean_field(self, logits):
        dim = -1
        probs = F.softmax(logits, dim)
        ndim = [1] * probs.dim()
        ndim[dim] = self.num_levels
        probs *= (
            torch
            .arange(self.num_levels, dtype=torch.float32, device=probs.device)
            .view(*ndim))
        return probs.sum(dim)

    def sample(self, logits):
        *lead_dims, softmax_dim = logits.shape
        probs = torch.softmax(logits, -1).view(-1, softmax_dim)
        samples = torch.multinomial(probs, 1)
        samples = samples.view(*lead_dims)
        return self.dequantize(samples)

    def loss(self, logits, targets):
        assert logits.size()[:4] == targets.size()
        logits = logits.permute(0, 4, 1, 2, 3)
        loss = F.cross_entropy(logits, targets.long(), reduction='none')
        return loss


class SoftmaxQuantizer(BaseDataQuantizer):
    def __init__(self, levels=[0.0, 0.25, 0.5, 0.75], **kwargs):
        super(SoftmaxQuantizer, self).__init__(**kwargs)
        assert levels == sorted(levels), "Levels should be sorted"
        self.register_buffer('levels', torch.tensor(levels))
        self.num_levels = len(levels)

    def forward(self, x):
        _, targets = torch.min((x.unsqueeze(-1) - self.levels)**2, -1)
        return self.dequantize(targets), targets

    def dequantize(self, q):
        return self.levels[q]

    def mean_field(self, logits):
        dim = -1
        probs = F.softmax(logits, dim)
        ndim = [1] * probs.dim()
        ndim[dim] = self.num_levels
        probs *= self.levels.view(*ndim)
        return probs.sum(dim)

    def sample(self, logits):
        *lead_dims, softmax_dim = logits.shape
        probs = torch.softmax(logits, -1).view(-1, softmax_dim)
        samples = torch.multinomial(probs, 1)
        samples = samples.view(*lead_dims)
        return self.dequantize(samples)

    def loss(self, logits, targets):
        assert logits.size()[:4] == targets.size()
        logits = logits.permute(0, 4, 1, 2, 3)
        loss = F.cross_entropy(logits, targets.long(), reduction='none')
        return loss


class BinaryXEntropy(BaseDataQuantizer):
    def __init__(self, **kwargs):
        super(BinaryXEntropy, self).__init__(**kwargs)

    def forward(self, x):
        assert x.min() >= 0.0
        assert x.max() <= 1.0
        return x, x

    def dequantize(self, x):
        return x

    def mean_field(self, logits):
        logits = safe_squeeze(logits, -1)
        return torch.sigmoid(logits)

    def sample(self, logits):
        logits = safe_squeeze(logits, -1)
        probs = torch.sigmoid(logits)
        return (torch.rand_like(probs) < probs
                ).float()

    def loss(self, logits, targets):
        logits = safe_squeeze(logits, -1)
        assert logits.size() == targets.size()
        return F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')


class L1Loss(BaseDataQuantizer):
    def __init__(self, **kwargs):
        super(L1Loss, self).__init__(**kwargs)

    def forward(self, x):
        return x, x

    def dequantize(self, x):
        return x

    def mean_field(self, logits):
        logits = safe_squeeze(logits, -1)
        return logits

    def sample(self, logits):
        logits = safe_squeeze(logits, -1)
        return Laplace(logits, 1.0).sample()

    def loss(self, logits, targets):
        logits = safe_squeeze(logits, -1)
        assert logits.size() == targets.size(), f"{logits.size()} != {targets.size()}"
        return F.l1_loss(logits, targets, reduction='none')


class L2Loss(BaseDataQuantizer):
    def __init__(self, **kwargs):
        super(L2Loss, self).__init__(**kwargs)

    def forward(self, x):
        return x, x

    def dequantize(self, x):
        return x

    def mean_field(self, logits):
        logits = safe_squeeze(logits, -1)
        return logits

    def sample(self, logits):
        logits = safe_squeeze(logits, -1)
        return Normal(logits, 1.0).sample()

    def loss(self, logits, targets):
        logits = safe_squeeze(logits, -1)
        assert logits.size() == targets.size()
        return F.mse_loss(logits, targets, reduction='none')


class NormalMeanScaleLoss(BaseDataQuantizer):
    def __init__(self, **kwargs):
        super(NormalMeanScaleLoss, self).__init__(**kwargs)
        self.num_levels = 2

    def forward(self, x):
        return x, x

    def dequantize(self, x):
        return x

    def mean_field(self, logits):
        return logits[:, :, :, :, 0]

    def _get_normal(self, logits):
        loc, scale = logits.chunk(2, dim=-1)
        loc = safe_squeeze(loc, -1)
        scale = torch.exp(safe_squeeze(scale, -1))
        return Normal(loc, scale)

    def sample(self, logits):
        return self._get_normal(logits).sample()

    def loss(self, logits, targets):
        assert logits.size()[:-1] == targets.size()
        return -self._get_normal(logits).log_prob(targets)


class PerceptualLoss(BaseDataQuantizer):
    def __init__(self, layers=6):
        super(PerceptualLoss, self).__init__()
        self.vgg = tvmodels.vgg16(pretrained=True).features[:layers]
        self.vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return x, x

    def dequantize(self, x):
        return x

    def mean_field(self, logits):
        logits = safe_squeeze(logits, -1)
        return logits

    def sample(self, logits):
        return safe_squeeze(logits, -1)

    def loss(self, logits, targets):
        logits = safe_squeeze(logits, -1)
        logits = logits.permute(0, 3, 2, 1)
        B, C, H, W = logits.shape
        logits = logits.expand(B, 3, H, W)
        targets = targets.permute(0, 3, 2, 1)
        targets = targets.expand(B, 3, H, W)
        return F.l1_loss(self.vgg(logits * 2 - 1), self.vgg(targets * 2 - 1),
                reduction='none')
