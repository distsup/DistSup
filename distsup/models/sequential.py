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

import sys
import math
import torch

from distsup import utils
from distsup import aligner
from distsup.models import base
from distsup.models.adversarial import Adversarial
from distsup.modules.predictors import GlobalPredictor


class SimpleCTCModel(base.Model):
    def __init__(self, encoder, num_classes=None,
                 allow_too_long_transcripts=False,
                 adv_size=0,
                 **kwargs):
        super(SimpleCTCModel, self).__init__(**kwargs)
        if num_classes is None:
            num_classes = self.dataloader.metadata['targets']['num_categories']
        self.encoder = utils.construct_from_kwargs(encoder)
        self.projection = torch.nn.Linear(
            self.encoder.output_dim, num_classes)
        self.ctc = torch.nn.CTCLoss(
            reduction='sum', zero_infinity=allow_too_long_transcripts)
        self.adversarial = None

        if adv_size != 0:
            self.adversarial = Adversarial(
                    GlobalPredictor(
                        self.encoder.output_dim,
                        adv_size, aggreg=10),
                    mode='maxent')

    def forward(self, features, features_len):
        # encoded: (bs x t x 1 x f)
        encoded, encoded_lens = self.encoder(features, features_len)

        if self.adversarial is not None:
            self.adversarial(encoded)
        log_probs = self.projection(encoded)
        log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)
        # log_probs: (bs x t x 1 x nc)
        return log_probs, encoded_lens

    def minibatch_loss(self, batch):
        # from distsup.utils import ptvsd; ptvsd()
        log_probs, log_prob_lens = self(
            batch['features'], batch['features_len'])
        targets = batch['targets'].int()
        targets_len = batch['targets_len']

        # log_probs: (bs x t x 1 x nc) -> (t x bs x nc)
        log_probs = utils.safe_squeeze(log_probs, 2).permute(1, 0, 2)
        loss = self.ctc(log_probs, targets,
                        log_prob_lens, targets_len) / log_prob_lens.size(0)
        decodes = utils.greedy_ctc_decode(log_probs, log_prob_lens)
        cer = utils.error_rate(
            decodes,
            [t[:tl] for t, tl in zip(targets.to('cpu').numpy(), targets_len)])
        details = {
                'cer': torch.tensor(cer),
                'main_loss': loss,
        }
        if self.adversarial is not None:
            friend_loss, adv_loss, adv_details = self.adversarial.loss(batch['spkid'])
            loss = loss + friend_loss# + adv_loss
            details['adv_friend_loss'] = friend_loss
            details['adv_adv_loss'] = adv_loss
            details['adv_acc'] = adv_details['acc']
        return loss, details

    def batch_to_device(self, batch, device):
        return {k: v.to(device) if isinstance(v, torch.Tensor)
                and not k.endswith('_len') else v
                for k, v in batch.items()}

    def evaluate(self, batches):
        tot_loss = 0
        tot_errs = 0
        tot_examples = 0
        tot_len_targets = 0
        for batch in batches:
            loss, stats = self.minibatch_loss(batch)
            num_examples = len(batch['features_len'])
            tot_loss += loss * num_examples
            tot_examples += num_examples
            len_targets = batch['targets_len'].sum().item()
            tot_errs += stats['cer'] * len_targets
            tot_len_targets += len_targets
        return {'loss': tot_loss / tot_examples,
                'cer': tot_errs / tot_len_targets}


class CTCModel(base.Model):
    def __init__(self, encoder,
                 num_classes=-1,
                 allow_too_long_transcripts=False,
                 alignment_name="",
                 forced_alignment=False,
                 verbose=0, **kwargs):
        super(CTCModel, self).__init__(**kwargs)

        # Determine number of classes for output layer
        alternativeNumClasses = len(self.dataset.alphabet)
        if alternativeNumClasses > 0 and alternativeNumClasses != num_classes:
            print ("CTCModel __init__() override yaml num_classes (" + str(num_classes) + \
                    ") with dataset alphabet num_classes (" + str(alternativeNumClasses) + ")" )
            num_classes = len(self.dataset.alphabet)
        assert num_classes > 0

        # Determine from .yaml if we need to produce an output path (alignment)
        # As we may need alignments for MNIST and ScribbleLens, we put the aligner in the model, not dataset
        self.aligner = None
        if alignment_name != "":
            if forced_alignment:
                self.aligner = aligner.ForcedAligner(alignment_name, self.dataset.alphabet)
            else:
                self.aligner = aligner.Aligner(alignment_name, self.dataset.alphabet)

        self.encoder = utils.construct_from_kwargs(encoder)
        self.verbose = verbose
        self.projection = torch.nn.Linear(
            self.encoder.output_dim, num_classes)
        self.ctc = torch.nn.CTCLoss(
            blank=0, reduction='sum', zero_infinity=allow_too_long_transcripts)

    '''
        Input:
            - features     is a Torch tensor of [1, seqLen, imgHeight==32, batchSize ]
            - features_len is a Torch tensor of [1 x batchSize] with seqLen
        Output:
            - log_probs is a Torch tensor of [ ~~seqLen/8, batchSize , nOutputCLasses]
    '''
    def forward(self, features, features_len):
        encoded, encoded_lens = self.encoder(features, features_len)
        log_probs = self.projection(encoded)
        log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)
        return log_probs, encoded_lens

    def minibatch_loss(self, batch):
        # Call forward() on this model
        log_probs, log_prob_lens = self(
            batch['features'], batch['features_len'])
        targets = batch['targets'].int()
        targets_len = batch['targets_len']
        # log_probs: (bs x t x 1 x nc) -> (t x bs x nc)
        log_probs = utils.safe_squeeze(log_probs, 2).permute(1, 0, 2)
        loss = self.ctc(log_probs, targets,
                        log_prob_lens, targets_len) / log_prob_lens.size(0)

        # 'decodes'           is a Python list of the sequence of best path tokens, per sample, no blanks
        # 'decodesWithBlanks' is the same but keeps the blanks for path generation and processing
        # 'log_probs' shape is [ maxLengthDecodeSequences, batchSize, num columns]
        decodes, decodesWithBlanks = utils.greedy_ctc_decode(log_probs, log_prob_lens, return_raw=True)

        szLongestProbSequence = log_prob_lens[0].item()
        batchSize = log_probs.shape[1]

        assert len(decodesWithBlanks[0]) == szLongestProbSequence
        assert len(decodesWithBlanks[0]) == log_probs.shape[0]
        assert szLongestProbSequence == log_probs.shape[0]

        # Pretty print of paths, strings and meanings
        # Also, write path to output file if requested.
        self.dataset.decode(self.aligner,
            decodesWithBlanks, decodes,
            log_probs, log_prob_lens,
            targets, targets_len,
            batch,
            self.verbose)

        # Calculate Levenshtein character (or label) error-rate, on clean strings
        cer = utils.error_rate(
            decodes,
            [t[:tl] for t, tl in zip(targets.to('cpu'), targets_len)])

        return loss, {'cer': torch.tensor(cer)}

    '''
        Compared to evaluate(), we assume there are no targets available.
        We also omit the CTC align.
        Given a model, you can call decode() and produce a symbol sequence as result.
        Sometimes, this is called a forward-only() or recognition-only pass.
    '''
    def decode(self, batch):
        # Call forward() on this model
        log_probs, log_prob_lens = self(
            batch['features'], batch['features_len'])
        # (bs x t x 1 x nc) --> (t x bs x nc)
        log_probs = utils.safe_squeeze(log_probs, 2).permute(1, 0, 2)
        # 'decodes'           is a Python list of the sequence of best path tokens, per sample, no blanks
        # 'decodesWithBlanks' is the same but keeps the blanks for path generation and processing
        # 'log_probs' shape is [ maxLengthDecodeSequences, batchSize, num columns]
        decodes, decodesWithBlanks = utils.greedy_ctc_decode(log_probs, log_prob_lens, return_raw=True)

        szLongestProbSequence = log_prob_lens[0].item()
        batchSize = log_probs.shape[1]

        assert len(decodesWithBlanks[0]) == szLongestProbSequence
        assert len(decodesWithBlanks[0]) == log_probs.shape[0]
        assert szLongestProbSequence == log_probs.shape[0]

        try:
            # Some datasets are fully transcribed. Use if available.
            targets = batch['targets'].int()
            targets_len = batch['targets_len']
        except:
            # We have no targets and therefore run a forward() only recognition.
            targets = None
            targets_len = None

        # Pretty print of paths, strings and meanings
        self.dataset.decode(self.aligner,
            decodesWithBlanks, decodes,
            log_probs, log_prob_lens,
            targets, targets_len,
            batch,
            self.verbose)

    def batch_to_device(self, batch, device):
        return {k: v.to(device) if isinstance(v, torch.Tensor) and not k.endswith('_len') else v
                for k, v in batch.items()}

    '''
    See "What size test set gives good error rate estimates? "
    I.Guyon, R.Schwartz, J.Makhoul, V.Vapnik
    PAMI 20/1 Jan 1998

    Compute a +/- signifance on test/train error rates
    n = nCorpusEntries
    p = recognition error-rate in [%] (like, 5.0 == 5%, NOT 0.05)
    D = significant difference in [%]
    Calculate D = (10.0 * p ) / sqrt( n )

    Uncertainty "D" is basically the minimal difference between 2 test error-rates
    before saying there is significant difference.
    '''
    def computeSignificantErrorDifference(self, nTokens_ , errorRate_):
        D = 100.0
        if nTokens_ > 0:
            D = ((10.0 * errorRate_) / math.sqrt( nTokens_ ))
        return D

    def evaluate(self, batches):
        tot_loss = 0
        tot_errs = 0
        tot_examples = 0
        tot_len_targets = 0
        for batch in batches:
            # Get loss and absolute error count via 'cer' key
            loss, stats = self.minibatch_loss(batch)

            batchCerPerc = stats['cer'] # in range [0..100.]
            batchCer = batchCerPerc * 0.01   # in range [ 0.. 1.]

            num_examples = len(batch['features_len'])
            tot_loss += loss * num_examples
            tot_examples += num_examples
            len_targets = batch['targets_len'].sum().item()
            tot_errs += batchCer * len_targets
            tot_len_targets += len_targets

        totalCerPercentage  = (tot_errs * 100.0) / tot_len_targets

        # uncertaintyCer in context of test/dev/evaluate on the whole dataset. Not on train batch.
        uncertaintyCer = self.computeSignificantErrorDifference(tot_len_targets, totalCerPercentage)
        print ("Evaluated a dev set of nSamples = " + str(tot_len_targets) + \
            " cer[%] = " + str(totalCerPercentage) + \
            " +/- uncertainty[%] " + str(uncertaintyCer)  )

        return {'loss': tot_loss / tot_examples,
                'cer': totalCerPercentage } # , 'confidence': uncertaintyCer}
