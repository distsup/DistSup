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

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from distsup import utils
from distsup.logger import DefaultTensorLogger
from distsup.modules import convolutional, wavenet
from distsup.models.adversarial import Adversarial
from distsup.models.base import Model
from distsup.models.streamtokenizer import StreamTokenizerNet
from distsup.modules.aux_modules import attach_auxiliary
from distsup.modules.bottlenecks import VQBottleneck
from distsup.modules.predictors import FramewisePredictor, GlobalPredictor
from distsup.modules.conditioning import CondNorm2d

logger = DefaultTensorLogger()


def vqvae_nanxin(input_height=28, codebook_size=50, **kwargs):
    encoder = [
        dict(in_channels=1,
             out_channels=256,
             padding=1,
             kernel_size=3,
             stride=2,
             bias=False),
        dict(in_channels=256,
             out_channels=256,
             kernel_size=3,
             padding=1,
             stride=2,
             bias=False),
        dict(in_channels=256, out_channels=64, padding=2, kernel_size=5, bias=True),
    ]
    decoder = [
        dict(in_channels=64, out_channels=256, padding=2, kernel_size=5, bias=False),
        dict(in_channels=256,
             out_channels=256,
             kernel_size=3,
             padding=1,
             stride=2,
             bias=False),
        dict(in_channels=256,
             out_channels=1,
             kernel_size=3,
             stride=2,
             output_padding=1,
             bias=True),
    ]
    return VQVAE(encoder, decoder, input_height=input_height, aggreg_stride=1,
            codebook_size=codebook_size, **kwargs)


class VQVAE(StreamTokenizerNet):
    """
    An image encoder-decoder model with a quantized bottleneck.

    Args:
        encoder (list of dicts): specification of convolutions for the
            encoder. BNs and ReLUs are added automatically except for the
            next layer.
        decoder (list of dicts): specification of convolutions for the
            decoder. BNs and ReLUs are added automatically except for the
            next layer. ConvTranspose2d are used.
        aggreg (int): how many spatial cells to aggregate as input for the
            quantization layer. Right now this has to be a multiple of the
            height of the latent variable since we have temporal models in
            mind, but this could be relaxed.
        codebook_size (int): how many quantized codes are learnable
        aggreg_stride (int or None): stride when aggregating cells in the
            width dimension. If set to None, they will be aggregated in non
            overlapping windows.
    """

    def __init__(self,
                 encoder,
                 decoder,
                 codebook_size,
                 input_height,
                 aggreg_stride=1,
                 adversarial_size=0,
                 adv_penalty=1e-9,
                 with_framewise_probe=False,
                 adv_class_embedding_size=128,
                 **kwargs):
        super(VQVAE, self).__init__(**kwargs)
        self.stride = aggreg_stride
        self.adv_penalty = adv_penalty
        self.indices = None

        self.encoder = []
        for conv in encoder[:-1]:
            self.encoder += [
                nn.Conv2d(**conv),
                nn.BatchNorm2d(conv['out_channels']),
                nn.ReLU(inplace=True)
            ]

        self.encoder.append(nn.Conv2d(**encoder[-1]))
        self.encoder = nn.Sequential(*self.encoder)

        self._compute_align_params()

        assert input_height % self.align_upsampling == 0, (
                "The height of the input image ({}) must be divisible by {}"
                ).format(input_height, self.align_upsampling)
        hidden_height = input_height // self.align_upsampling

        d = self.encoder[-1].out_channels * aggreg_stride * hidden_height
        self.vq = VQBottleneck(d, d, codebook_size, dim=1)

        self.char_pred = None
        if with_framewise_probe:
            self.char_pred = attach_auxiliary(
                    self.vq,
                    FramewisePredictor(d, len(self.dataset.alphabet), aggreg=2),
                    bp_to_main=False)

        self.adversarial = None
        if adversarial_size != 0:
            self.adversarial = Adversarial(
                    GlobalPredictor(
                        self.encoder[-1].out_channels * hidden_height,
                        adversarial_size, time_reduce='max',
                        aggreg=5), mode='reverse')

        self.align_upsampling *= aggreg_stride

        self.embs = nn.Embedding(len(self.dataset.alphabet),
                adv_class_embedding_size)
        self.decoder = []
        for conv in decoder[:-1]:
            self.decoder += [
                nn.ConvTranspose2d(**conv),
                CondNorm2d(conv['out_channels'], adv_class_embedding_size),
                nn.ReLU(inplace=True)
            ]
        self.decoder.append(nn.ConvTranspose2d(**decoder[-1]))
        self.decoder.append(nn.Sigmoid())
        self.decoder = nn.ModuleList(self.decoder)

        self.add_probes()

        self.apply(utils.conv_weights_xavier_init)

    def _compute_align_params(self):
        self.align_upsampling = 1
        self.align_offset = 0

        for module in self.encoder.children():
            if isinstance(module, nn.Conv2d):
                self.align_offset += (module.kernel_size[-1] - 1) // 2 - module.padding[1]
                self.align_upsampling *= module.stride[1]

            elif isinstance(module, (nn.BatchNorm2d, nn.ReLU)):
                logging.debug(f'Layer of type {module.__class__.__name__} '
                              f'is assumed not to change the data rate.')

            else:
                raise NotImplementedError(f'The rate handling of module {module.__class__.__name__} '
                                          f'has not been implemented.'
                                          f'If this module affects the data rate, '
                                          f'handle the way in which the alignment changes.'
                                          f'If not, add it to the previous case of this if statement.')

        return

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, y):
        y_emb = self.embs(y)

        for m in self.decoder:
            if isinstance(m, CondNorm2d):
                z = m(z, y_emb)
            else:
                z = m(z)
        return z

    def forward(self, x, y):
        z = self.encode(x)

        aggreg_width = self.stride
        b, zc, zh, zw = z.shape

        z = F.unfold(z, (zh, aggreg_width),
                     stride=(1, self.stride or aggreg_width))

        if self.adversarial is not None:
            self.adversarial(z)
        zq, _, details = self.vq(z)
        self.indices = details['indices']

        zq = F.fold(zq,
                    output_size=(zh, zw),
                    kernel_size=(zh, aggreg_width),
                    stride=(1, self.stride or aggreg_width))


        x2 = self.decode(zq, y)

        return x2, details['indices']

    def align_tokens_to_features(self, batch, tokens):
        token2 = tokens.repeat_interleave(self.align_upsampling, dim=1)
        token2 = token2[:, :batch['features'].shape[1]]
        return token2

    def minibatch_loss_and_tokens(self, batch):
        x = batch['features']
        x = x.permute(0, 3, 2, 1)
        mask = utils.get_mask2d(batch['features_len'], x)
        if x.shape[3] % self.align_upsampling != 0:
            os = torch.zeros(
                    *x.shape[0:3],
                    self.align_upsampling - x.shape[3] % self.align_upsampling,
                    device=x.device)
            x = torch.cat([x, os], dim=3)
            mask = torch.cat([mask, os], dim=3)

        if 'adversarial' in batch:
            x2, indices = self(x * 2 - 1, batch['adversarial'])
        else:
            x2, indices = self(x * 2 - 1, torch.zeros(x.shape[0],
                device=x.device).long())

        logger.log_images('orig', x[:3])
        logger.log_images('recs', x2[:3])

        main_loss = self.loss(x, x2, mask)
        details = {
                'recon_loss': main_loss,
        }

        if self.char_pred is not None and 'alignment' in batch:
            char_loss, char_details = self.char_pred.loss(x, batch['alignment'])
            details['char_loss'] = char_loss
            details['char_acc'] = char_details['acc']
            main_loss += char_loss

        if self.adversarial is not None and 'adversarial' in batch:
            friend_loss, advloss, adv_details = self.adversarial.loss(batch['adversarial'])
            main_loss = main_loss + friend_loss + self.adv_penalty * advloss
            details['adversarial_loss'] = advloss
            details['adversarial_friendly_loss'] = friend_loss
            details['adversarial_acc'] = adv_details['acc']

        return main_loss, details, indices

    def loss(self, x, x2, mask):
        recon = F.l1_loss(x2 * mask, x)
        return recon
