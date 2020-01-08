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
from distsup.modules.misc import Exp, GatedTransition, Combiner
from distsup.logger import DefaultTensorLogger
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from collections import defaultdict
import torch.distributions as dist
from itertools import tee
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

logger = DefaultTensorLogger()


class FramePredictor(nn.Module):
    def __init__(self, input_dim, output_dim, aggreg=3):
        super(FramePredictor, self).__init__()
        # self.pred = nn.Conv1d(input_dim, output_dim, kernel_size=aggreg)
        self.kernel_size = 3
        self.pred = nn.Sequential(nn.Conv1d(input_dim, 128, self.kernel_size),
                                  nn.BatchNorm1d(128),
                                  nn.ReLU(),
                                  nn.Conv1d(128, output_dim, 1))

    def forward(self, encoding):
        # bsz, time, channel --> bsz, channel, time
        encoding = encoding.permute(0, 2, 1)
        encoding = F.pad(encoding, (0, self.kernel_size -1, 0, 0))
        out_conv = self.pred(encoding)
        return out_conv.permute(0, 2, 1)

    def plot(self, x, pred):
        import matplotlib.pyplot as plt
        fig = plt.Figure()
        (top, bottom) = fig.subplots(2)
        top.margins(y=0)
        bottom.margins(y=0)
        fig.tight_layout(h_pad=0)
        top.imshow(x[0, 0].cpu())
        bottom.imshow(pred[0].cpu().transpose(0, 1), aspect='auto',
                      interpolation='nearest')
        top.get_xaxis().set_visible(False)
        return fig

    def loss(self, x, labels, x_lens, mask):
        out = self(x)
        labels = labels.long()
        pred_labels = out.argmax(dim=2)
        correct = 0
        total = 0
        for i in range(x.size(0)):
            seq_length = x_lens[i].cpu().detach().item()
            correct += (pred_labels[i, :seq_length] == labels[i, :seq_length]).sum().item()
            total += seq_length
        acc = correct/total
        loss = F.cross_entropy(out.contiguous().view(-1, out.size(-1)),
                               labels.contiguous().view(-1), reduction='none') * mask.contiguous().view(-1)
        # if logger.is_currently_logging():
        #     logger.log_mpl_figure('framewise debug',
        #                           self.plot(x, F.softmax(out.detach(), dim=-1)))
        loss = loss.mean()
        details = {
            'loss': loss,
            'acc': acc,
            'out_seq': pred_labels.detach(),
        }
        return loss, details


class DMM(Model):
    # A Deep Markov Model inspired by [1]
    # Krishnan, R. G., Shalit, U., & Sontag, D. (2017, February).
    # Structured inference networks for nonlinear state space models.
    # In Thirty-First AAAI Conference on Artificial Intelligence.
    # Model Description
    # Encoder: LSTM (no subsampling)
    # Decoder: 2 layer FeedForward NN
    # First order markov chain on the latent space
    # Markov transitions are modeled using a Feed Forward Neural Network
    # Evaluation:
    # a) Frame level phone prediction accuracy
    # b) ELBO
    # c) KL
    # d) Number of active units in the latent code (checking for posterior collapse)
    def __init__(self, z_size=32, feat_size=81, hidden_size=256, transition_size=128,
                 **kwargs):
        super(DMM, self).__init__(**kwargs)
        self.z_size = z_size
        self.x_size = feat_size
        self.h_size = hidden_size
        self.q_z_nn = self.make_encoder()
        self.p_x_nn, self.p_x_mu, self.p_x_scale = self.make_decoder()
        self.transition_fn = GatedTransition(self.z_size, transition_size)
        self.frame_predictor = FramePredictor(self.z_size, 44)
        self.combiner = Combiner(self.z_size, self.h_size)
        self.z_q_0 = nn.Parameter(torch.zeros(self.z_size))

    def make_encoder(self):
        q_z_nn = nn.LSTM(self.x_size, self.h_size, batch_first=True)
        return q_z_nn

    def make_decoder(self):
        p_x_nn = nn.Sequential(nn.Linear(self.z_size, self.h_size),
                               nn.ReLU(),
                               nn.Linear(self.h_size, self.h_size),
                               nn.ReLU())
        p_x_mu = nn.Linear(self.h_size, self.x_size)
        p_x_scale = nn.Sequential(nn.Linear(self.h_size, self.x_size),
                                  Exp())
        return p_x_nn, p_x_mu, p_x_scale

    def sample_z_posterior(self, x, x_lens):
        bsz, T_max, _ = x.size()
        x_rev = utils.reverse_sequences(x, x_lens)
        x_rev = pack_padded_sequence(x_rev, x_lens, batch_first=True)
        rnn_output, _ = self.q_z_nn(x_rev)
        rnn_output = utils.pad_and_reverse(rnn_output, x_lens)
        z_prev = self.z_q_0.expand(bsz, self.z_q_0.size(0))
        mus = []
        scales = []
        zs = []
        for t in range(1, T_max + 1):
            z_mu, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])
            # reparam sample
            z = dist.Normal(z_mu, z_scale).rsample()
            mus.append(z_mu)
            scales.append(z_scale)
            zs.append(z)
            z_prev = z

        mus = torch.stack(mus, dim=1)
        scales = torch.stack(scales, dim=1)
        zs = torch.stack(zs, dim=1)

        return zs, (mus, scales)

    def transition(self, z):
        bsz, t, _ = z.size()
        # start with zero and one
        z_mu, z_scale = self.transition_fn(z.view(bsz * t, -1))
        z_mu_0 = torch.zeros(bsz, 1, z_mu.size(-1)).to(z.device.type)
        z_scale_0 = torch.ones(bsz, 1, z_scale.size(-1)).to(z.device.type)
        z_mu = z_mu.contiguous().view(bsz, t, -1)
        z_scale = z_scale.contiguous().view(bsz, t, -1)
        z_mu = torch.cat([z_mu_0, z_mu[:, :-1]],
                         dim=1)  # bsz, t, dim
        z_scale = torch.cat([z_scale_0, z_scale[:, :-1]], dim=1)
        return z_mu, z_scale

    def likelihood(self, z):
        bsz, time, _ = z.size()
        context = self.p_x_nn(z)
        loc = self.p_x_mu(context)
        scale = self.p_x_scale(context)
        return dist.Normal(loc, scale)

    def forward(self, x, x_lens):
        z, z_variational_params = self.sample_z_posterior(x, x_lens)
        _, z_prior_params = self.transition(z)
        llk = self.likelihood(z)
        return llk, z_variational_params, z_prior_params, z

    @staticmethod
    def kl_temporal(mean_q, scale_q, mean_prior, scale_prior, mask):
        """temporal KL, Equation 11 from:
        Krishnan, Rahul G., Uri Shalit, and David Sontag. "Structured inference networks
        for nonlinear state space models." Thirty-First AAAI Conference on Artificial Intelligence. 2017."""
        # batch mean of kl
        var_q = scale_q**2
        var_prior = scale_prior**2
        diff_mu = mean_prior - mean_q
        kl = 0.5 * (torch.log(var_prior) - torch.log(var_q) - 1. + var_q / var_prior + diff_mu ** 2 / var_prior)
        return kl*mask.unsqueeze(-1)

    def compute_elbo(self, x, llk_fn, z_variational_params, z_prior_params, mask, kl_mult=1.):
        llk = llk_fn.log_prob(x)*mask.unsqueeze(-1)
        llk = llk.sum((1, 2))
        kl = DMM.kl_temporal(z_variational_params[0], z_variational_params[1],
                             z_prior_params[0], z_prior_params[1], mask)
        kl_sum = kl.sum((1, 2))
        elbo = llk - kl_sum
        annealed_elbo = llk - kl_mult*kl_sum
        total_time_steps_batch = float(torch.sum(mask))
        return annealed_elbo, {"cond_ll": llk.sum()/total_time_steps_batch,
                               "elbo": elbo.sum()/total_time_steps_batch,
                               "kl": kl_sum.sum()/total_time_steps_batch}

    def compute_prediction_loss(self, z, labels, seq_lens, mask):
        loss, details = self.frame_predictor.loss(z, labels, seq_lens, mask)
        return loss, details

    def minibatch_loss(self, batch):
        # do not take the first and second order derivatives of energies
        x = batch["features"][..., 0]
        x_lens = batch["features_len"]
        time_aligned_labels = batch["alignment"]
        batch_mask = utils.get_mini_batch_mask(x, x_lens).to(x.device.type)
        llk_fn, z_variational_params, z_prior_params, z = self(x, x_lens)
        annealed_elbo, info = self.compute_elbo(x, llk_fn, z_variational_params, z_prior_params, batch_mask)
        pred_loss, details = self.compute_prediction_loss(z.detach(), time_aligned_labels, x_lens, batch_mask)
        info["frame_acc"] = torch.tensor(details["acc"])
        return annealed_elbo.mean().mul_(-1)+pred_loss, info

    def evaluate(self, batches):
        info_all = defaultdict(list)
        correct = 0
        total = 0
        z_means = []
        for batch in batches:
            x = batch["features"][..., 0]
            x_lens = batch["features_len"]
            time_aligned_labels = batch["alignment"]
            batch_mask = utils.get_mini_batch_mask(x, x_lens).to(x.device.type)
            llk_fn, z_variational_params, z_prior_params, z = self(x, x_lens)
            for i in range(x.size(0)):
                z_means.append(z_variational_params[0][i, :x_lens[i].cpu().item()].contiguous()
                               .view(-1, z.size(-1)))
            _, info = self.compute_elbo(x, llk_fn, z_variational_params, z_prior_params, batch_mask)
            scores = self.frame_predictor(z).contiguous()
            predictions = scores.max(-1)[1]
            for i in range(x.size(0)):
                seq_length = x_lens[i].cpu().detach().item()
                correct += (predictions[i, :seq_length] == time_aligned_labels[i, :seq_length]).sum().item()
                total += seq_length
            for k, v in info.items():
                info_all[k].append(v)

        number_of_active_units, _ = utils.calc_au(z_means)
        accuracy = correct / total
        info_dataset = dict([(k, torch.mean(torch.stack(v))) for k, v in info_all.items()])
        info_dataset["frame_acc"] = accuracy
        info_dataset["active_units"] = number_of_active_units
        return info_dataset

class FDMM(Model):
    # https://arxiv.org/pdf/1803.02991.pdf
    # http://people.csail.mit.edu/sameerk/papers/dmm_camera.pdf
    # A Deep Markov Model inspired by [1]
    # Krishnan, R. G., Shalit, U., & Sontag, D. (2017, February).
    # Structured inference networks for nonlinear state space models.
    # In Thirty-First AAAI Conference on Artificial Intelligence.
    # Model Description
    # Encoder: LSTM (no subsampling)
    # Decoder: 2 layer FeedForward NN
    # First order markov chain on the latent space
    # Markov transitions are modeled using a Feed Forward Neural Network
    # Evaluation:
    # a) Frame level phone prediction accuracy
    # b) ELBO
    # c) KL
    # d) Number of active units in the latent code (checking for posterior collapse)
    def __init__(self, z_size=32, f_size=32, feat_size=81, hidden_size=256, transition_size=128,
                 **kwargs):
        super(FDMM, self).__init__(**kwargs)
        self.z_size = z_size
        self.x_size = feat_size
        self.h_size = hidden_size
        self.q_z_nn = self.make_encoder(self.h_size)
        self.p_x_nn, self.p_x_mu, self.p_x_scale = self.make_decoder(z_size + f_size)
        self.transition_fn = GatedTransition(self.z_size, transition_size)
        self.frame_predictor = FramePredictor(self.z_size, 44)
        self.combiner = Combiner(self.z_size, self.h_size + f_size)
        self.z_q_0 = nn.Parameter(torch.zeros(self.z_size))
        self.q_f_nn = self.make_encoder(f_size, bidirectional=True)

    def make_encoder(self, size, bidirectional=False):
        q_z_nn = nn.LSTM(self.x_size, size, batch_first=True, bidirectional=bidirectional)
        return q_z_nn

    def make_decoder(self, size):
        p_x_nn = nn.Sequential(nn.Linear(size, self.h_size),
                               nn.ReLU(),
                               nn.Linear(self.h_size, self.h_size),
                               nn.ReLU())
        p_x_mu = nn.Linear(self.h_size, self.x_size)
        p_x_scale = nn.Sequential(nn.Linear(self.h_size, self.x_size),
                                  Exp())
        return p_x_nn, p_x_mu, p_x_scale

    def sample_f_posterior(self, x, x_lens):
        x_pack = pack_padded_sequence(x, x_lens, batch_first=True)
        rnn_output, _ = self.q_f_nn(x_pack)
        rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)
        mask = (x_lens-1).unsqueeze(1).unsqueeze(2).expand(-1, x.size(1), rnn_output.size(2))
        q_f = rnn_output.gather(1, mask.long())[:, 0, :]
        q_mean, q_logscale = torch.chunk(q_f, 2, dim=-1)
        q_scale = torch.exp(q_logscale)
        f = dist.Normal(q_mean, q_scale).rsample()
        return f, (q_mean, q_scale)

    def sample_z_posterior(self, x, x_lens, f):
        bsz, T_max, _ = x.size()
        x_rev = utils.reverse_sequences(x, x_lens)
        x_rev = pack_padded_sequence(x_rev, x_lens, batch_first=True)
        rnn_output, _ = self.q_z_nn(x_rev)
        rnn_output = utils.pad_and_reverse(rnn_output, x_lens)
        z_prev = self.z_q_0.expand(bsz, self.z_q_0.size(0))
        mus = []
        scales = []
        zs = []
        for t in range(1, T_max + 1):
            z_mu, z_scale = self.combiner(z_prev, torch.cat((rnn_output[:, t - 1, :], f), dim=-1))
            # reparam sample
            z = dist.Normal(z_mu, z_scale).rsample()
            mus.append(z_mu)
            scales.append(z_scale)
            zs.append(z)
            z_prev = z

        mus = torch.stack(mus, dim=1)
        scales = torch.stack(scales, dim=1)
        zs = torch.stack(zs, dim=1)

        return zs, (mus, scales)

    def transition(self, z):
        bsz, t, _ = z.size()
        # start with zero and one
        z_mu, z_scale = self.transition_fn(z.view(bsz * t, -1))
        z_mu_0 = torch.zeros(bsz, 1, z_mu.size(-1)).to(z.device.type)
        z_scale_0 = torch.ones(bsz, 1, z_scale.size(-1)).to(z.device.type)
        z_mu = z_mu.contiguous().view(bsz, t, -1)
        z_scale = z_scale.contiguous().view(bsz, t, -1)
        z_mu = torch.cat([z_mu_0, z_mu[:, :-1]],
                         dim=1)  # bsz, t, dim
        z_scale = torch.cat([z_scale_0, z_scale[:, :-1]], dim=1)
        return z_mu, z_scale

    def likelihood(self, z, f):
        bsz, time, dim = z.size()
        context = self.p_x_nn(torch.cat((z, f.unsqueeze(1).expand(bsz, time, dim)), dim=-1))
        loc = self.p_x_mu(context)
        scale = self.p_x_scale(context)
        return dist.Normal(loc, scale)

    def forward(self, x, x_lens):
        f, f_variational_params = self.sample_f_posterior(x, x_lens)
        z, z_variational_params = self.sample_z_posterior(x, x_lens, f)
        _, z_prior_params = self.transition(z)
        llk = self.likelihood(z, f)
        return llk, z_variational_params, z_prior_params, z, f_variational_params, f

    @staticmethod
    def kl_temporal(mean_q, scale_q, mean_prior, scale_prior, mask=None):
        """temporal KL, Equation 11 from:
        Krishnan, Rahul G., Uri Shalit, and David Sontag. "Structured inference networks
        for nonlinear state space models." Thirty-First AAAI Conference on Artificial Intelligence. 2017."""
        # batch mean of kl
        var_q = scale_q**2
        var_prior = scale_prior**2
        diff_mu = mean_prior - mean_q
        kl = 0.5 * (torch.log(var_prior) - torch.log(var_q) - 1. + var_q / var_prior + diff_mu ** 2 / var_prior)
        return kl*mask.unsqueeze(-1) if mask is not None else kl

    def compute_elbo(self, x, llk_fn, z_variational_params, z_prior_params, f_variational_params, mask, kl_mult=1.):
        llk = llk_fn.log_prob(x)*mask.unsqueeze(-1)
        llk = llk.sum((1, 2))
        kl = FDMM.kl_temporal(z_variational_params[0], z_variational_params[1],
                             z_prior_params[0], z_prior_params[1], mask)
        kl_sum = kl.sum((1, 2)) + FDMM.kl_temporal(f_variational_params[0], f_variational_params[1],
                                                  f_variational_params[0] * 0, f_variational_params[1] * 0 + 1).sum(dim=-1)
        elbo = llk - kl_sum
        annealed_elbo = llk - kl_mult*kl_sum
        total_time_steps_batch = float(torch.sum(mask))
        return annealed_elbo, {"cond_ll": llk.sum()/total_time_steps_batch,
                               "elbo": elbo.sum()/total_time_steps_batch,
                               "kl": kl_sum.sum()/total_time_steps_batch}

    def compute_prediction_loss(self, z, labels, seq_lens, mask):
        loss, details = self.frame_predictor.loss(z, labels, seq_lens, mask)
        return loss, details

    def minibatch_loss(self, batch):
        # do not take the first and second order derivatives of energies
        x = batch["features"][..., 0]
        x_lens = batch["features_len"]
        time_aligned_labels = batch["alignment"]
        batch_mask = utils.get_mini_batch_mask(x, x_lens).to(x.device.type)
        llk_fn, z_variational_params, z_prior_params, z, f_variational_params, _ = self(x, x_lens)
        annealed_elbo, info = self.compute_elbo(x, llk_fn, z_variational_params, z_prior_params, f_variational_params, batch_mask)
        pred_loss, details = self.compute_prediction_loss(z.detach(), time_aligned_labels, x_lens, batch_mask)
        info["frame_acc"] = torch.tensor(details["acc"])
        return annealed_elbo.mean().mul_(-1)+pred_loss, info

    def evaluate(self, batches):
        info_all = defaultdict(list)
        correct = 0
        total = 0
        z_means = []
        for batch in batches:
            x = batch["features"][..., 0]
            x_lens = batch["features_len"]
            time_aligned_labels = batch["alignment"]
            batch_mask = utils.get_mini_batch_mask(x, x_lens).to(x.device.type)
            llk_fn, z_variational_params, z_prior_params, z, f_variational_params, _ = self(x, x_lens)
            for i in range(x.size(0)):
                z_means.append(z_variational_params[0][i, :x_lens[i].cpu().item()].contiguous()
                               .view(-1, z.size(-1)))
            _, info = self.compute_elbo(x, llk_fn, z_variational_params, z_prior_params, f_variational_params, batch_mask)
            scores = self.frame_predictor(z).contiguous()
            predictions = scores.max(-1)[1]
            for i in range(x.size(0)):
                seq_length = x_lens[i].cpu().detach().item()
                correct += (predictions[i, :seq_length] == time_aligned_labels[i, :seq_length]).sum().item()
                total += seq_length
            for k, v in info.items():
                info_all[k].append(v)

        number_of_active_units, _ = utils.calc_au(z_means)
        accuracy = correct / total
        info_dataset = dict([(k, torch.mean(torch.stack(v))) for k, v in info_all.items()])
        info_dataset["frame_acc"] = accuracy
        info_dataset["active_units"] = number_of_active_units
        return info_dataset
