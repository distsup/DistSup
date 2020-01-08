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
import torch.nn as nn
import torch.nn.functional as F

from distsup import utils
from distsup.configuration import Globals
from distsup.models import base


class MNISTMlp(base.Model):
    def minibatch_loss(self, batch):
        logits = self(batch['image']) #softmax output
        preds = logits.argmax(dim=1, keepdim=True) #highest for each batch
        num_classes = np.arange(logits.size(1)).reshape(1,1,logits.size(1))
        #batch alignment contains chunk-len alignment
        #check the most seen for each batch
        ali_num = batch['alignment'].cpu().numpy()[:,:,np.newaxis]
        #produce a 3D mask (size #batch,#chunk,#numclasses), then add the mask for each row
        #producing a 2D matrix (size #batch, #numclasses)
        #then select the argmax
        maj_ali=torch.from_numpy((ali_num==num_classes).sum(axis=1).argmax(axis=1))
        #print("preds :", preds)
        #print("maj_ali :", maj_ali)
        stats = {'cer (%)': preds.cpu().ne(maj_ali.view_as(preds)).sum().float() / logits.size(0)*100}
        #print("Stats :", stats)
        return F.nll_loss(logits.cpu(), maj_ali), stats

    def evaluate(self, batches):
        tot_loss = 0
        tot_errs = 0
        tot_examples = 0
        for batch in batches:
            num_examples = batch['image'].size(0)
            loss, stats = self.minibatch_loss(batch)
            tot_examples += num_examples
            tot_loss += loss * num_examples
            tot_errs += stats['cer (%)'] * num_examples

        return {'loss': tot_loss / tot_examples,
                'cer (%)': tot_errs / tot_examples}


class MLP(MNISTMlp):
    def __init__(self, num_inputs, num_classes,
                 hidden_dims=(), activation='ReLU', **kwargs):
        super(MLP, self).__init__(**kwargs)
        num_inputs = np.prod(num_inputs)
        self.net, num_inputs = utils.get_mlp(
            num_inputs, hidden_dims, activation)
        self.net.add_module(f'proj', nn.Linear(num_inputs, num_classes))
        self.net.add_module(f'sm', nn.LogSoftmax(dim=-1))

    def forward(self, x):
        #print("x:", x, "x.size:", x.size(0))
        #print(x.size(1))
        #print(x.size(2))
        x = x.view(x.size(0), -1)
        return self.net(x)
