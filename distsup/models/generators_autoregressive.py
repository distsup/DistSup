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

'''Autoregressive models: Column-wise WaveNet ad PixelCNNs.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from distsup import utils
from distsup.logger import DefaultTensorLogger
from distsup.models import base
from distsup.modules import misc
from distsup.modules import reconstructors

logger = DefaultTensorLogger()


class Autoregressive2D(base.Model):
    def __init__(self,
                 condition_on_alignment=False,
                 num_alignment_classes=11,  # For MNIST
                 image_height=28,
                 cond_mixer=dict(
                     class_name=misc.IdentityForgetKWargs),
                 reconstructor=[
                     dict(class_name=reconstructors.ColumnGatedPixelCNN,),
                 ],
                 **kwargs):
        super(Autoregressive2D, self).__init__(**kwargs)
        self.condition_on_alignment = condition_on_alignment
        if self.condition_on_alignment:
            self.num_alignment_classes = num_alignment_classes
            self.cond_mixer = utils.construct_from_kwargs(cond_mixer)
            self.cond_mixer.eval()
            mixer_out_channels = self.cond_mixer(
                torch.empty((1, 100, 1, num_alignment_classes))).size(3)
            self.cond_channels = ({
                'cond_dim': mixer_out_channels, 'reduction_factor': 1},)
        else:
            self.cond_channels = ()

        rec_params = {'image_height': image_height,
                      'cond_channels': cond_channels_spec}
        # Compatibility with single-reconstructor checkpoints
        if 'class_name' in reconstructor:
            self.reconstructor = utils.construct_from_kwargs(
                reconstructor, additional_parameters=rec_params)
            self.reconstructors = {'': self.reconstructor}
        else:
            self.reconstructors = nn.ModuleDict({
                name: utils.construct_from_kwargs(
                    rec, additional_parameters=rec_params)
                for name,rec in reconstructor.items()})

    def _prepare_cond(self, alignment):
        if self.condition_on_alignment:
            alignment = F.one_hot(
                alignment.long(), self.num_alignment_classes
                ).float().unsqueeze(2).contiguous()
            alignment = self.cond_mixer(alignment)
            return (alignment, )
        else:
            return ()

    def minibatch_loss(self, batch):
        # from distsup.utils import ptvsd; ptvsd()
        conds = self._prepare_cond(batch['alignment'])

        details = {}
        for name, rec in self.reconstructors.items():
            inputs, targets = rec.get_inputs_and_targets(
                batch['image'], batch.get('image_lens'))
            logits = rec(inputs, conds)
            loss = rec.loss(logits, targets)
            loss = loss.mean()  # nats/pix

            name = '_' + name if name else name
            details[f'rec{name}_loss'] = loss

            if logger.is_currently_logging():
                def log_img(name, img):
                    logger.log_images(name, img.permute(0, 2, 1, 3))
                log_img(f'x{name}', inputs[:4])
                log_img(f'p{name}', rec.get_mean_field_preds(logits[:4]))
                if not self.training:
                    priming = inputs[:3].repeat_interleave(2, dim=0)
                    if 'alignment' in batch:
                        alignment = batch['alignment'
                                          ][:3].repeat_interleave(2, dim=0)
                    else:
                        alignment = None
                    logger.log_mpl_figure(
                        f'gen_samples{name}',
                        rec.plot_debug_samples(
                            priming, self._prepare_cond(alignment)))

        tot_loss = sum(details.values())
        details['rec_loss'] = tot_loss
        return tot_loss, {'rec_loss': loss}


class MNISTWaveNet(Autoregressive2D):
    def __init__(self,
                 reconstructor=dict(
                     class_name=reconstructors.ColumnwiseWaveNet,),
                 **kwargs):
        super(MNISTWaveNet, self).__init__(reconstructor=reconstructor,
                                           **kwargs)
