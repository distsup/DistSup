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

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from distsup.models.base import Model
import distsup.models.simple as models
from distsup.modules.predictors import GlobalPredictor, FramewisePredictor


class Adversarial(nn.Module):
    """
    Create an adversarial learner to be attached to a layer in order to force
    a main learner to generate features invariant to some criterion. The
    adversarial learner will try to learn some category from the features,
    while the main learner will learn to produce features that won't allow such
    discrimination.
    """

    def __init__(self, clf, beta, mode='reverse'):
        """
        Constructor.

        Args:
            clf (nn.Module): the model to use as adversarial trainer.
        """
        super(Adversarial, self).__init__()
        self.clf = clf
        self.mode = mode
        assert mode in ['reverse', 'maxent']
        self.beta = beta

    def forward(self, output, features_len, targets_len):
        if isinstance(output, (list, tuple)):
            output = output[0]
        # Make the adversarial learn, and not the main model.
        self.train()
        self.unfreeze()
        friendly = self.clf(output.detach(), features_len, targets_len)

        # Make the main model learn to confuse the adversarial learner.
        self.eval()
        self.freeze()
        adversarial = self.clf(output, features_len, targets_len)
        return friendly, adversarial

    def loss(self, features, y, features_len=None, targets_len=None):
        friendly, adversarial = self(self.input, features_len, targets_len)
        friendloss = F.cross_entropy(friendly, y)

        if self.mode == 'reverse':
            advloss = -F.cross_entropy(adversarial, y)
        elif self.mode == 'maxent':
            advloss = -self.entropy(adversarial)

        return friendloss + self.beta * advloss, {
            "acc": torch.mean(
                (friendly.detach().argmax(dim=1) == y).float()),
            "friendloss": friendloss,
            "advloss": advloss
        }


    def entropy(self, x):
        log_p = F.log_softmax(x, dim=1)
        p = log_p.exp()
        return torch.mean(-torch.sum(p * log_p, dim=1), dim=0)

    def freeze(self):
        for p in self.clf.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.clf.parameters():
            p.requires_grad = True


class GlobalAdversary(Adversarial):
    def __init__(self, input_dim, output_dim, beta, kernel_size=1, mode='maxent',
            time_reduce='avg'):
        super(GlobalAdversary, self).__init__(
                GlobalPredictor(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    aggreg=kernel_size,
                    time_reduce=time_reduce),
                beta,
                mode=mode)

# Warning: I'm untested
class LinearAdversary(Adversarial):
    def __init__(self, input_dim, output_dim, beta, aggreg=1,
            use_two_layer_predictor=False, mode='reverse'):
        super(LinearAdversary, self).__init__(
                FramewisePredictor(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    aggreg=aggreg,
                    use_two_layer_predictor=use_two_layer_predictor),
                beta,
                mode=mode)

