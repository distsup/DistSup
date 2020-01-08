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

import torch.nn as nn
import torch


class Model(nn.Module):
    """
    A trainable model.

    Args:
        dataloader: the training dataloader. It is automatically provided
            by the YAML instantiator.
    """
    def __init__(self, dataloader=None, **kwargs):
        super(Model, self).__init__(**kwargs)
        if dataloader:
            self.dataloader = dataloader
            self.dataset = dataloader.dataset
        else:
            self.dataloader = None
            self.dataset = None

    def get_parameters_for_optimizer(self, with_codebook=True):
        if with_codebook:
            return self.parameters()
        else:
            def _parameters():
                for name, p in self.named_parameters():
                    if 'bottleneck.embedding' not in name:
                        yield p
            return _parameters()

    def batch_to_device(self, batch, device):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def minibatch_loss(self, batch):
        raise NotImplementedError

    def evaluate(self, batches):
        tot_loss = 0
        tot_examples = 0
        for batch in batches:
            num_examples = len(next(iter(batch.values())))
            loss, _ = self.minibatch_loss(batch)
            tot_examples += num_examples
            tot_loss += loss * num_examples
        return {'loss': tot_loss / tot_examples}
