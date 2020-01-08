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

import distsup.utils as utils
from distsup.models.base import Model
from distsup.modules.bottlenecks import SelfOrganizingMapBottleNeck
from distsup.logger import DefaultTensorLogger
from distsup.models import streamtokenizer
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

logger = DefaultTensorLogger()


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.contiguous().view(*self.shape)


class SOMVAEMnistSeq(streamtokenizer.StreamTokenizerNet):
    # MNIST Sequence Data
    def __init__(self, image_height=28, h_size=256, num_tokens=[8, 8], latent_size=64, alpha=0.25, beta=1.,
                 **kwargs):
        super(SOMVAEMnistSeq, self).__init__(**kwargs)
        self.in_size = image_height
        self.h_size = h_size
        self.latent_size = latent_size
        self.encoder = nn.LSTM(image_height, h_size, batch_first=True)
        self.som_bn = SelfOrganizingMapBottleNeck(h_size, latent_size, num_tokens, dim=-1)
        # do not feed the ground truth
        self.p_x_nn, self.p_x_loc = self.make_decoder()
        self.alpha = alpha
        self.beta = beta

        self.add_probes()

    def make_decoder(self):
        p_x_nn = nn.LSTM(self.latent_size, self.h_size, batch_first=True)
        p_x_loc = nn.Sequential(nn.Linear(self.h_size, self.in_size),
                                nn.Sigmoid())
        return p_x_nn, p_x_loc

    def encode(self, x):
        out, _ = self.encoder(x)
        loss, info = self.som_bn(out)
        return loss, info

    def decode(self, z):
        # z: bsz, time, dim
        context, _ = self.p_x_nn(z)
        return self.p_x_loc(context)

    def forward(self, x):
        input_shape = x.size()[:-1]
        # self-organizing map bottleneck
        vq_loss, info = self.encode(x)
        # an LSTM decoder
        reco_e = self.decode(info["ze"].contiguous().view(*input_shape, -1))
        reco_q = self.decode(info["zq"].contiguous().view(*input_shape, -1))
        return reco_e, reco_q, vq_loss, info["indices"]

    def align_tokens_to_features(self, batch, tokens):
        # No downsampling in our case
        return utils.safe_squeeze(tokens, 1)

    def minibatch_loss_and_tokens(self, batch):
        x = batch['features']  # bsz, time, dim, 1
        x = x.squeeze(-1)
        # check inputs
        assert x.min() == 0.0
        assert x.max() == 1.0
        input_shape = x.size()[:-1]
        reco_e, reco_q, vq_loss, nodes = self(x)
        # should be the log probability of x under the bernoulli distn.
        reco_loss_e = F.mse_loss(reco_e, x)
        reco_loss_q = F.mse_loss(reco_q, x)
        loss = reco_loss_e + reco_loss_q + vq_loss

        details = {"reco_e": reco_loss_e,
                   "reco_q": reco_loss_q,
                   "vq_loss": vq_loss}

        return loss, details, nodes.contiguous().view(*input_shape)

    def minibatch_loss(self, batch):
        loss, stats, *_ = self.minibatch_loss_and_tokens(batch)
        return loss, stats

    def evaluate(self, batches):
        tot_examples = 0.
        tot_loss = 0.
        tot_errs = 0.

        alis_es = []
        alis_gt = []
        alis_lens = []
        for batch in batches:
            first_field = next(iter(batch.values()))
            num_examples = len(first_field)
            loss, stats, tokens = self.minibatch_loss_and_tokens(batch)
            feat_len = batch['features_len']
            alis_lens.append(feat_len)

            # the tokens should match the rate of the alignment
            ali_es = self.align_tokens_to_features(batch, tokens)
            assert ((ali_es.shape[0] == batch['features'].shape[0]))
            assert ((ali_es.shape[1] == batch['features'].shape[1]))
            alis_es.append(ali_es)

            if 'alignment' in batch:
                ali_gt = batch['alignment']
                ali_len = batch['alignment_len']

                assert ((ali_len == feat_len).all())
                alis_gt.append(ali_gt)

            tot_examples += num_examples
            tot_loss += loss * num_examples
            tot_errs += stats.get('err', np.nan) * num_examples

        all_scores = {'loss': tot_loss / tot_examples,
                      'err': tot_errs / tot_examples}

        alis_es = self._unpad_and_concat(alis_es, alis_lens)
        alis_gt = self._unpad_and_concat(alis_gt, alis_lens) if len(alis_gt) else None

        scores_to_compute = [('', lambda x: x)]
        if alis_gt is not None and self.pad_symbol is not None:
            not_pad = (alis_gt != self.pad_symbol)
            scores_to_compute.append(('nonpad_', lambda x: x[not_pad]))

        for prefix, ali_filter in scores_to_compute:
            es = ali_filter(alis_es)

            if alis_gt is not None:
                gt = ali_filter(alis_gt)

                mapping_scores = self._mapping_metrics(gt, es, prefix=prefix)
                all_scores.update(mapping_scores)

                clustering_scores = self._clustering_metrics(gt, es, prefix=prefix)
                all_scores.update(clustering_scores)

            perplexity_scores = self._perplexity_metrics(es, prefix=prefix)
            all_scores.update(perplexity_scores)

        return all_scores


class SOMVAEScribbleLens(Model):
    # TODO
    def __init__(self):
        ...

    def forward(self):
        ...
