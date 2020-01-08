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

import torch.nn as nn
import torch.nn.functional as F

from distsup import utils
from distsup.configuration import Globals
from distsup.models import base


class ClassifierNet(base.Model):
    def minibatch_loss(self, batch):
        logits = self(batch['features'])
        preds = logits.argmax(dim=1, keepdim=True)
        stats = {'err': preds.ne(batch['targets'].view_as(preds)
                                 ).sum().float() / logits.size(0)
                 }
        return F.nll_loss(logits, batch['targets']), stats

    def evaluate(self, batches):
        tot_loss = 0
        tot_errs = 0
        tot_examples = 0
        for batch in batches:
            num_examples = batch['features'].size(0)
            loss, stats = self.minibatch_loss(batch)
            tot_examples += num_examples
            tot_loss += loss * num_examples
            tot_errs += stats['err'] * num_examples

        return {'loss': tot_loss / tot_examples,
                'err': tot_errs / tot_examples}


class MLP(ClassifierNet):
    def __init__(self, num_inputs, num_classes,
                 hidden_dims=(), activation='ReLU', **kwargs):
        super(MLP, self).__init__(**kwargs)
        num_inputs = np.prod(num_inputs)
        self.net, num_inputs = utils.get_mlp(
            num_inputs, hidden_dims, activation)
        self.net.add_module(f'proj', nn.Linear(num_inputs, num_classes))
        self.net.add_module(f'sm', nn.LogSoftmax(dim=-1))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class MNISTConvNet(ClassifierNet):
    def __init__(self, num_classes, **kwargs):
        super(MNISTConvNet, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
