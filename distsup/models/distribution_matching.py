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
import torch.nn as nn
import torch.nn.functional as F

from distsup import utils
from distsup.models import base
from distsup.configuration import Globals
from distsup import configuration, utils
from distsup.alphabet import Alphabet
import distsup.checkpoints
import distsup.models.simple

import numpy as np
import argparse
import pickle
import gzip

import pdb


def calculate_odm_loss(batch_log_probs, lm_probs, lm_probs_ngrams):
    """Computes output distribution matching loss.
    
    Args:
        batch_log_probs: log probabilities over characters of a single batch
        lm_probs (1D tensor of shape (num_grams,)): language model *log* 
            probabilities that we are trying to match 
        lm_probs_ngrams (2D tensor of shape (num_ngrams, lm_order)):  
            ngrams (character IDs) that correspond to the log probs in lm_probs
    """
    batch_size = len(batch_log_probs)
    observed_lm_log_probs = torch.zeros((batch_size, len(lm_probs)))
    if Globals.cuda:
        observed_lm_log_probs = observed_lm_log_probs.to('cuda')
    n = batch_log_probs.shape[1]
    for i in range(n):
        observed_lm_log_probs += batch_log_probs[:, i, lm_probs_ngrams[:, i]]

    mean_observed_lm_log_probs = observed_lm_log_probs.logsumexp(
      dim=0) - math.log(batch_size)
    
    return -1.0 * (lm_probs * mean_observed_lm_log_probs).sum()

class DistributionMatchingClassifier(distsup.models.simple.ClassifierNet):
    """
    The classifier for distribution matching should not have a softmax
    layer.
    """

    def init_output_biases(self, biases):
        """Sets the biases of the last layer to the given
        values and also sets the weights of the last layer to 0
        """
        raise Error("not implemented")
    
    

class MLP(DistributionMatchingClassifier):

    def __init__(self, num_inputs, num_classes,
                 hidden_dims=(), activation='ReLU', **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        num_inputs_prod = np.prod(num_inputs)
        self.net, num_inputs = utils.get_mlp(
            num_inputs_prod, hidden_dims, activation)
        
        self.net.add_module(f'proj', nn.Linear(num_inputs, num_classes))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

    def init_output_biases(self, biases):
        self.net[-1].bias.data = biases
        self.net[-1].weight.data *= 0


class MNISTConvNet(DistributionMatchingClassifier):
    def __init__(self, num_classes, **kwargs):
        super(MNISTConvNet, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def init_output_biases(self, biases):
        self.fc2.bias.data = biases
        self.fc2.weight.data *= 0


class LinearLR(torch.optim.lr_scheduler.LambdaLR):

    def __init__(self, optimizer, num_epochs, final_scale):
        super(LinearLR, self).__init__(optimizer, lr_lambda=lambda epoch: final_scale ** (epoch / num_epochs))


def read_scoring_map(filename):
    """
    Reads a scoring map like the one used for TIMIT phone-level scoring.
    Each phone in a single line is mapped to the last phone in the line
    for scoring purposes.
    """
    result = {}
    for l in open(filename):
        ss = l.split()
        for p in ss[0:-1]:
            result[p] = ss[-1]
    return result

class DistributionMatchingModel(base.Model):
    """
    Distribution matching model.
    """
    
    def __init__(self, classifier,
                 temperature, frame_pair_batch_size=10000,
                 scoring_map=None, lambda_fs=0.5, verbose=0, **kwargs):
        """
        Args:
            classifier: classifier that maps batches of feature chunks
                to output logits. See DistributionMatchingClassifier.
            temperature: temperature used in the distribution matching estimation
            frame_pair_batch_size: batch size used for the secondary 
                (frame similarity) loss.
            scoring_map: (optional) used for mapping output characters to
                other characters, for scoring purposes (needed for the
                stupid TIMIT evaluation scheme).      
        """
        super(DistributionMatchingModel, self).__init__(**kwargs)
        self.classifier =  utils.construct_from_kwargs(classifier)       
        self.alphabet = self.dataset.alphabet
        self.lm_order = self.dataset.order

        self.lm_probs = self.dataset.lm_probs
        self.lm_probs_ngrams = self.dataset.lm_probs_ngrams

        if Globals.cuda:
            self.lm_probs = self.lm_probs.to('cuda')
            self.lm_probs_ngrams = self.lm_probs_ngrams.to('cuda')
  
        self.temperature = temperature
        self.verbose = verbose
        self.lambda_fs = lambda_fs
        self.pair_batches = self.dataset.get_frame_pair_iter(frame_pair_batch_size)

        self.classifier.init_output_biases(self.dataset.output_frequencies.log())

        self.scoring_map = None
        if scoring_map is not None:
            self.scoring_map = read_scoring_map(scoring_map)
                      

    def forward(self, sampled_features):
        logits = self.classifier(sampled_features)
        return logits
       

    def minibatch_loss(self, batch):
        input_features = batch['sampled_frames']
        orig_shape = input_features.shape
        # sampled featurer are provided  in a shape bsz * ngram_order * ...
        # flatten and unflatten the 1st two dims
        logits = self(input_features.view((orig_shape[0] * orig_shape[1],) + orig_shape[2:]))
        logits = logits.view((orig_shape[0], orig_shape[1], len(self.dataset.outputs)))    
        
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)
        
        #import pdb; pdb.set_trace()
        loss_odm = calculate_odm_loss(log_probs, self.lm_probs, self.lm_probs_ngrams)

        pair_batch = next(self.pair_batches)
        if Globals.cuda:
            pair_batch = self.batch_to_device(pair_batch, 'cuda')

        paired_features = pair_batch["feature_pairs"]
        orig_shape = paired_features.shape
        # paired features are provided  in a shape bsz * 2 * ...
        # flatten and unflatten the 1st two dims
        pair_logits = self(paired_features.view((orig_shape[0] * orig_shape[1],) + orig_shape[2:]))
        pair_logits = pair_logits.view((orig_shape[0], orig_shape[1], len(self.dataset.outputs))) 
        pair_log_probs = F.log_softmax(pair_logits / self.temperature, dim=-1)
        loss_fs = F.mse_loss(pair_log_probs[:, 0], pair_log_probs[:, 1], reduction='mean')
        loss = loss_odm + self.lambda_fs * loss_fs        
        return loss, {'odm_loss': loss_odm, 'fs_loss': loss_fs, 'total_loss': loss} 


    def evaluate(self, batches):
        tot_loss = 0
        tot_errs = 0
        tot_errs_mapped = 0
        tot_examples = 0
        for batch in batches:
            num_examples = batch[self.dataset.feature_field].size(0)
            targets = batch["target"]
            for i in range(len(targets)):
                targets[i] = self.dataset.alphabet_to_output[targets[i].item()]
            logits = self(batch[self.dataset.feature_field])
            assert logits.shape[0] == targets.shape[0]
            assert logits.shape[1] == len(self.dataset.output_frequencies)
            tot_loss += F.cross_entropy(logits, targets, reduction='sum').item()
            predictions = torch.max(logits, 1)[1].view(targets.size()).data
            tot_errs += (predictions != targets.data).sum().item()
            if self.scoring_map is not None:
                for i in range(len(predictions)):
                    predicted_phoneme = self.alphabet.idx2ch(predictions[i].item())
                    target_phoneme = self.alphabet.idx2ch(targets.data[i].item())
                    if self.scoring_map.get(predicted_phoneme, predicted_phoneme) != self.scoring_map.get(target_phoneme, target_phoneme):
                        tot_errs_mapped += 1
                        
            tot_examples += num_examples

        return {'loss': tot_loss / tot_examples,
                'err': tot_errs / tot_examples,
                'mapped_err': tot_errs_mapped / tot_examples}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_yaml', type=str, help='Model to load')
    parser.add_argument('features_pkl', type=str, help="Compute probs for features in pkl file")
    
    parser.add_argument('--output-loglikes', type=str, help="Output Kaldi ark file with log likelihoods")
    parser.add_argument('--use-prior', type=str, help="When outputting predictions, divide by the given prior")
    
    parser.add_argument('--output-prior', type=str, help="Output a prior over the model outputs, based on the features")

    args = parser.parse_args()
    
    config, model = distsup.checkpoints.get_config_and_model(config=args.model_yaml)
    
    model = model.to('cpu')
    model.eval()
    if args.output_prior:
        likelihood_sums = None
        with torch.no_grad():
            all_features = pickle.load(gzip.open(args.features_pkl))
            for utt_id, features in all_features:
                logits = model.forward(torch.tensor([features]))
                loglikes = F.softmax(logits, dim=2)
                if likelihood_sums is not None:
                    likelihood_sums += loglikes[0].mean(dim=0)
                else:
                    likelihood_sums = loglikes[0].mean(dim=0)
            with open(args.output_prior, "w") as f:
                print(" ".join([str(v.item()) for v in likelihood_sums/len(all_features)]), file=f)
                    
            
    elif args.output_loglikes:
        prior = None
        if args.use_prior:            
            prior = torch.tensor([float(s) for s in open(args.use_prior).readline().split()])
        with torch.no_grad():
            from kaldiio import WriteHelper
            with WriteHelper(f'ark:{args.output_loglikes}') as writer:
                all_features = pickle.load(gzip.open(args.features_pkl))
                for utt_id, features in all_features:
                    logits = model.forward(torch.tensor([features]))
                    loglikes = F.log_softmax(logits, dim=2)
                    if prior is not None:
                        loglikes -= prior.log()
                    writer(utt_id, loglikes[0].detach().numpy())
        
    
