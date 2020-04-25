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

from operator import mul
import random
import time
import numpy as np

from sklearn import cluster

import torch
import torch.nn as nn
import torch.distributions as distributions
from torch.autograd import Function
import torch.nn.functional as F

from distsup.configuration import Globals
from distsup.utils import construct_from_kwargs, get_mask1d, safe_squeeze
from distsup.logger import DefaultTensorLogger
from distsup.modules.encoders import Identity, Normalization
from distsup.modules.misc import *

logger = DefaultTensorLogger()


class NullBottleneck(nn.Module):
    def __init__(self, in_dim, latent_dim, log_input_norms=False, dim=-1):
        super(NullBottleneck, self).__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.projection = nn.Linear(in_dim, latent_dim)

    def forward(self, x, side_info=None, enc_len=None):
        """Project and sample x.

        Args:
            - x: tensor pof shape D1, in_dim, D2

        Returns tuple of
            - samples of shape D1, latent_dim, D2
            - KL of shape D1, 1, D2
            - dict of:
                  embeddings: mean values for latent code
        """
        del side_info  # unused
        del enc_len  # unused
        dim = self.dim
        needs_transpose = dim != -1 or dim != x.dim() - 1
        if needs_transpose:
            x = x.transpose(-1, dim)

        if self.log_input_norms:
            norms = torch.norm(x.view(-1, x.size(-1)).contiguous(), dim=1)
            logger.log_scalar('vq_input_norms_pre_bn/mean', torch.mean(norms))
            logger.log_scalar('vq_input_norms_pre_bn/std', torch.std(norms))

        z = self.projection(x)

        if self.log_input_norms:
            norms = torch.norm(z.view(-1, z.size(-1)).contiguous(), dim=1)
            logger.log_scalar('vq_input_norms_post_fc/mean', torch.mean(norms))
            logger.log_scalar('vq_input_norms_post_fc/std', torch.std(norms))

        if needs_transpose:
            z = z.transpose(-1, dim)
        # Logging here?
        return z, 0, {'embeddings': z, 'indices': None}


class VAEBottleneck(nn.Module):
    """
    Add a VAE bottleneck along a specified dimension.

    This layer combines a linear projection and sampling.

    It returns the generated samples, together with the KL computed
    per-latent vector.

    Args:
        - in_dim: dimensionality of the input
        - latent_dim: dimensionality of th latent code
        - dim: dimension (axis) to operate
    """
    def __init__(self, in_dim, latent_dim, dim=-1):
        # TODO:
        # - freebits
        # - proper KL normalization and logging
        super(VAEBottleneck, self).__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.projection = nn.Linear(in_dim, latent_dim * 2)

    def forward(self, x):
        """Project and sample x.

        Args:
            - x: tensor pof shape D1, in_dim, D2

        Returns tuple of
            - samples of shape D1, latent_dim, D2
            - KL of shape D1, 1, D2
            - dict of:
                  embeddings: mean values for latent code
        """
        dim = self.dim
        needs_transpose = dim != -1 or dim != x.dim() - 1
        if needs_transpose:
            x = x.transpose(-1, dim)
        projected = self.projection(x)
        mu, logvar = projected.split(self.latent_dim, -1)
        z = torch.exp(0.5 * logvar) * torch.randn_like(mu) + mu
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),
                              dim=-1, keepdim=True)
        if needs_transpose:
            z = z.transpose(-1, dim)
            kl = kl.transpose(-1, dim)
            mu = mu.transpose(-1, dim)
        # Logging here?
        return z, kl, {'embeddings': mu, 'indices': None}


class ReservoirSampler(nn.Module):
    def __init__(self, num_samples=1024):
        super(ReservoirSampler, self).__init__()
        self.n = num_samples
        self.ttot = 0
        self.register_buffer('buffer', None)
        self.reset()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        buffer_key = prefix + 'buffer'
        if buffer_key in state_dict:
            self.buffer = state_dict[buffer_key]
        return super(ReservoirSampler, self
                     )._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def reset(self):
        self.i = 0
        self.buffer = None

    def add(self, samples):
        self.ttot -= time.time()
        samples = samples.detach()
        if self.buffer is None:
            self.buffer = torch.empty(
                self.n, samples.size(-1), device=samples.device)
        buffer = self.buffer
        if self.i < self.n:
            slots = self.n - self.i
            add_samples = samples[:slots]
            samples = samples[slots:]
            buffer[self.i: self.i + len(add_samples)] = add_samples
            self.i += len(add_samples)
            if not len(samples):
                print(f"Res size {self.i}")
                self.ttot += time.time()
                return

        for s in samples:
            # warning, includes right end too.
            idx = random.randint(0, self.i)
            self.i += 1
            if idx < len(buffer):
                buffer[idx] = s
        self.ttot += time.time()

    def contents(self):
        return self.buffer[:self.i]


class VQBottleneck(nn.Module):
    def __init__(self, in_dim, latent_dim, num_tokens, dim=-1,
                 commitment=0.25, criterion='nearest',
                 criterion_eval='nearest',
                 criterion_kwargs={},
                 use_copy_through=False,
                 ignore_masked=False,
                 self_loop_bonus_reestimation_num_merges=25,
                 self_loop_bonus_reestimation_smoothing=0.99,
                 reestimation_reservoir_size=None,
                 reestimate_every_epochs=0,
                 reestimate_every_iters=0,
                 reestimate_every_iters_expansion=0,
                 reestimate_max_iters=0,
                 reestimate_max_epochs=0,
                 bottleneck_enforce_from_epoch=-1,
                 log_input_norms=False,
                 normalization='none',
                 normalization_nary=1,
                 normalization_set_affine=None,
                 input_projection=True,
                 ):
        super(VQBottleneck, self).__init__()

        # NOTE This might be Identity()
        self.batch_norm = Normalization(normalization, normalization_nary,
            in_dim, set_affine=normalization_set_affine)
        self.log_input_norms = log_input_norms
        # TODO Check if batch_norm processes the dimensions in the right order
        if input_projection:
            self.projection = nn.Linear(in_dim, latent_dim)
        else:
            assert in_dim == latent_dim, (
                f"No projection to bottleneck, input ({in_dim}) "
                f"and latent ({latent_dim}) dims dont agree")
            self.projection = Identity()

        self.embedding = nn.Embedding(num_tokens, latent_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.num_tokens = num_tokens
        self.dim = dim
        self.commitment = commitment
        self.use_copy_through = use_copy_through
        self.ignore_masked = ignore_masked
        self.criterion = criterion
        assert self.criterion != 'hmm_segmental'
        self.criterion_eval = criterion_eval

        if self.criterion_eval == 'hmm_segmental':
            self.greedy_hmm_matrices = self.get_greedy_hmm_matrices()

        # Used to track the self loop probability for segmental
        self.register_buffer('self_loop_bonus_estimate', torch.zeros(()))
        self.self_loop_bonus_reestimation_num_merges = self_loop_bonus_reestimation_num_merges
        self.self_loop_bonus_reestimation_smoothing = self_loop_bonus_reestimation_smoothing

        needs_global_info = criterion_kwargs.pop('needs_global_info', False)
        if needs_global_info:
            #self.adapt_fc = nn.Linear(emb_dim, latent_dim, bias=False)
            #self.adapt_fc.weight.data.fill_(0.0)
            self.adapt_bias = nn.Linear(256, latent_dim, bias=False)
            self.adapt_bias.weight.data.fill_(0.0)
            self.needs_global_info = True
        else:
            self.needs_global_info = False

        self.criterion_kwargs = criterion_kwargs
        self.bottleneck_enforce_from_epoch = bottleneck_enforce_from_epoch
        self.register_buffer('reestimation_data',
                             torch.tensor([
                                 Globals.epoch,
                                 Globals.current_iteration,
                                 reestimate_every_epochs,  # next reestimation epoch
                                 int(reestimate_every_iters),  # next reestimation iter
                                 1,  # Reestimation is operating
                                 ], dtype=torch.int32))
        if reestimation_reservoir_size:
            self.reestimation_reservoir = ReservoirSampler(
                reestimation_reservoir_size)
            self.reestimate_every_epochs = reestimate_every_epochs
            self.reestimate_last_epoch = Globals.epoch
            self.reestimate_every_iters = reestimate_every_iters
            self.reestimate_every_iters_expansion = reestimate_every_iters_expansion
            self.reestimate_max_epochs = reestimate_max_epochs
            self.reestimate_max_iters = reestimate_max_iters
            assert reestimate_every_epochs or reestimate_every_iters
        else:
            self.reestimation_reservoir = None

    def reestimate(self):
        #
        # When warming up, we keep the encodings from the last epoch and
        # reestimate just before new epoch starts.
        # When quantizing, we reestimate every number of epochs or iters given.
        #
        (last_epoch, last_iter, next_reest_epoch, next_reest_iter, is_operating
         ) = self.reestimation_data
        if not is_operating:
            print(f"Re-Disabling reestimation buffer")
            self.reestimation_reservoir = None
            return
        if self.bottleneck_enforce_from_epoch > 0:
            # Warmup
            if last_epoch == Globals.epoch:
                return
            # A new epoch has started:
            self.reestimation_data[0] = Globals.epoch
            if Globals.epoch < self.bottleneck_enforce_from_epoch:
                print("Reseting reservoir")
                self.reestimation_reservoir.reset()
                return
            else:
                # We will start quantizing soon, let it run
                pass
        else:
            # Normal operation
            if (self.reestimate_every_epochs and Globals.epoch < next_reest_epoch):
                return
            if (self.reestimate_every_iters and
                    Globals.current_iteration < next_reest_iter):
                return

        # Set the next reestimation iter.
        if self.reestimate_every_iters_expansion:
            next_reest_iter = (
                Globals.current_iteration * self.reestimate_every_iters) + 1
        else:
            next_reest_iter = (
                Globals.current_iteration + self.reestimate_every_iters)
        self.reestimation_data[:4] = torch.tensor([
            Globals.epoch,
            Globals.current_iteration,
            Globals.epoch + self.reestimate_every_epochs,
            next_reest_iter])

        if self.reestimation_reservoir.buffer is None:
            return
        tstart = time.time()
        num_clusters = self.embedding.weight.size(0)
        encodings = self.reestimation_reservoir.contents()
        if encodings.size(0) < num_clusters:
            print(f"Skipping reestimation, too few samples")
            return
        encodings = encodings.cpu().numpy()
        clustered, *_ = cluster.k_means(encodings, num_clusters)
        self.embedding.weight.data[
            ...] = torch.tensor(clustered).to(self.embedding.weight.device)
        self.reestimation_reservoir.reset()
        print(f"Done reestimating VQ embedings, took {time.time() - tstart}s")
        if ((self.reestimate_max_epochs and
             Globals.epoch > self.reestimate_max_epochs)
            or
            (self.reestimate_max_iters and
             Globals.current_iteration > self.reestimate_max_iter)):
            print(f"Disabling reestimation buffer")
            self.reestimation_data[4] = 0
            self.reestimation_reservoir = None

        self.post_reestim_hook()

    def pack_x(self, x, x_lens):
        if x_lens is None or self.ignore_masked:
            return x
        else:
            mask = get_mask1d(x_lens.to(x.device)
                              ).unsqueeze(-1).unsqueeze(-1) > 0
            x_sel = torch.masked_select(x, mask)
            x_sel = x_sel.view(mask.sum(), x.size(-1))
            return x_sel

    def unpack_x(self, x, x_lens):
        if x_lens is None or self.ignore_masked:
            return x
        else:
            x_seqs = x.split(tuple(x_lens))
            x = torch.nn.utils.rnn.pad_sequence(
                x_seqs, batch_first=True).unsqueeze(2)
            return x

    def codebook_train_hook(self, x_flat, codes_one_hot):
        pass

    def post_reestim_hook(self):
        pass

    def forward(self, x, side_info=None, enc_len=None):
        if self.reestimation_reservoir and self.training:
            self.reestimate()

        if self.log_input_norms:
            norms = torch.norm(x.view(-1, x.size(-1)).contiguous(), dim=1)
            logger.log_scalar('vq_input_norms_pre_bn/mean', torch.mean(norms))
            logger.log_scalar('vq_input_norms_pre_bn/std', torch.std(norms))

        dim = self.dim
        needs_transpose = dim != -1 or dim != x.dim() - 1
        if needs_transpose:
            x = x.transpose(-1, dim).contiguous()

        if self.batch_norm is not Identity:
            N, H, W, C = x.size()
            x = x.permute(0, 3, 1, 2).contiguous().view(N, C, -1)
            x = self.batch_norm(x)
            x = x.view(N, C, H, W).permute(0, 2, 3, 1).contiguous()

        if self.log_input_norms:
            norms = torch.norm(x.view(-1, x.size(-1)).contiguous(), dim=1)
            logger.log_scalar('vq_input_norms_post_bn/mean', torch.mean(norms))
            logger.log_scalar('vq_input_norms_post_bn/std', torch.std(norms))

        x = self.projection(x)

        if self.log_input_norms:
            norms = torch.norm(x.view(-1, x.size(-1)).contiguous(), dim=1)
            logger.log_scalar('vq_input_norms_post_fc/mean', torch.mean(norms))
            logger.log_scalar('vq_input_norms_post_fc/std', torch.std(norms))

        if self.needs_global_info:
            fc = 1.  # F.sigmoid(self.adapt_fc(emb)).unsqueeze(1).unsqueeze(1) + 1e-20
            bias = self.adapt_bias(side_info).unsqueeze(1).unsqueeze(1)
            x = (x - bias) / fc

        if self.training and self.reestimation_reservoir:
            self.reestimation_reservoir.add(
                self.pack_x(x, enc_len)
                    .view(-1, x.size(-1)).detach())

        if self.training or self.criterion == 'sparse':
            criterion = self.criterion
        else:
            criterion = self.criterion_eval
        if criterion == 'nearest' and not self.training:
            criterion_kwargs = {}
        else:
            criterion_kwargs = self.criterion_kwargs

        if Globals.epoch < self.bottleneck_enforce_from_epoch:
            print("Skipping quantization")
            codes = x
            indices = torch.zeros(x.shape[:-1] + (1,), device=x.device, dtype=torch.int64)
            values = indices
        elif criterion == 'hmm_segmental':
            assert not self.training
            codes, indices = self.quantize_hmm_segmental(x, enc_len)
        else:
            x = self.pack_x(x, enc_len)
            if criterion == 'sparse':
                assert not self.use_copy_through
                codes, indices = sparse_quantize(x, self.embedding.weight, self.commitment,
                                                 criterion_kwargs)
            else:
                ret = quantize(x, self.embedding.weight, self.commitment,
                               criterion, criterion_kwargs, self.use_copy_through)
                if len(ret) == 2:
                    codes, indices = ret
                else:
                    codes, indices, values = ret
                    self.update_self_loop_bonus(values)
            codes = self.unpack_x(codes, enc_len)
            indices = self.unpack_x(indices, enc_len)

        self.codebook_train_hook(
            x, F.one_hot(indices, self.embedding.weight.size(0)).float())

        self._log_code_usage(indices, criterion)
        if self.needs_global_info:
            codes = (codes * fc) + bias

        if needs_transpose:
            codes = codes.transpose(-1, dim)
            indices = indices.transpose(-1, dim)

        kl = (torch
              .empty(indices.shape, device=codes.device)
              .fill_(torch.log(torch.tensor(self.embedding.weight.size(0),
                                            dtype=torch.float32))))
        if self.log_input_norms:
            norms = torch.norm(codes.view(-1, codes.size(-1)).contiguous(), dim=1)
            logger.log_scalar('vq_input_norms_post_bottle/mean', torch.mean(norms))
            logger.log_scalar('vq_input_norms_post_bottle/std', torch.std(norms))

        if criterion != 'segmental':
            return codes, kl, {'indices': indices,
                               'embeddings': codes.detach(),
                               'pre_bn_acts': x.transpose(-1, dim)}
        else:
            return codes, kl, {'indices': indices,
                               'segmental_values': values,
                               'embeddings': codes.detach(),
                               'pre_bn_acts': x.transpose(-1, dim)}

    def _log_code_usage(self, indices, criterion):
        if not logger.is_currently_logging():
            return
        num_tokens = self.embedding.weight.size(0)
        code_freqs = torch.histc(
            indices.float(),
            bins=num_tokens, min=-0.5, max=num_tokens - 0.5
            ).float()
        count = np.prod(indices.size())
        if criterion != 'sparse':
            assert code_freqs.sum().item() == count
        code_freqs /= count
        entropy = distributions.Categorical(code_freqs).entropy()

        logger.log_scalar("vq_code_usage_frac",
                          entropy.item() / np.log(num_tokens))

    def get_greedy_hmm_matrices(self):
        proto_weights = self.embedding.weight
        N = proto_weights.shape[0]

        neg_inf = -1e20
        # Greedy matrix creation
        # state 0 is starting, state n + 1 refers to token N
        # each state has N outgoing edges
        states_mat = torch.arange(
            0, N + 1, dtype=torch.int64).view(1, N + 1).repeat((N + 1, 1)).unsqueeze(0)
        ilabels_mat = (states_mat.transpose(1, 2) - 1).clamp(0, N)
        weights_mat = torch.zeros_like(states_mat, dtype=torch.float)
        weights_mat[0, 0, :] = neg_inf
        terminal_mat = torch.zeros((N + 1, 1), dtype=torch.float32).unsqueeze(0)
        terminal_mat[0, 0] = neg_inf

        return [
            states_mat, ilabels_mat, weights_mat, terminal_mat
        ]

    def update_self_loop_bonus(self, values):
        self_loop_bonus = (
            values[-1] - values[-self.self_loop_bonus_reestimation_num_merges - 1]
        ) / self.self_loop_bonus_reestimation_num_merges
        self.self_loop_bonus_estimate *= (
            self.self_loop_bonus_reestimation_smoothing)
        self.self_loop_bonus_estimate += (
            (1.0 - self.self_loop_bonus_reestimation_smoothing) *
            self_loop_bonus
        )
        logger.log_scalar('self_loop_bonus', self_loop_bonus)
        logger.log_scalar('smooth_self_loop_bonus', self.self_loop_bonus_estimate)
        # print(f"SL_bonus: {self_loop_bonus} smooth: {self.self_loop_bonus_estimate}")

    def quantize_hmm_segmental(self, x, x_lens=None):
        from distsup.modules import fst_utils

        self.greedy_hmm_matrices = [m.to(x.device) for m in self.greedy_hmm_matrices]
        (states_mat, ilabels_mat, weights_mat, terminal_mat
         ) = self.greedy_hmm_matrices
        weights_mat_seg = weights_mat + torch.diag(
            torch.empty(weights_mat.size(-1), device=weights_mat.device
                        ).fill_(self.self_loop_bonus_estimate))
        seg_mats = [
            states_mat, ilabels_mat, weights_mat_seg, terminal_mat
        ]

        proto_weights = self.embedding.weight
        log_probs = -torch.cdist(x.detach().reshape(-1, proto_weights.shape[1]),
                                 proto_weights)
        log_probs = log_probs.view(x.shape[:-1] + (-1,)).requires_grad_(True)
        if x_lens is None:
            x_lens = torch.empty(
                log_probs.shape[0], dtype=torch.int64).fill_(log_probs.shape[1])
        with torch.enable_grad():
            loss = fst_utils.path_reduction(
                log_probs, x_lens, seg_mats, red_kind='viterbi',)
            loss.sum().backward()

        _, seg_idx = torch.max(log_probs.grad, dim=-1)
        quantized = self.embedding(seg_idx)
        seg_idx.unsqueeze_(-1)
        num_segments = float((np.diff(seg_idx.view(seg_idx.shape[:2]).cpu().numpy(),
                                      axis=1) != 0).sum())
        logger.log_scalar('hmm_segment_frac', num_segments / float(x_lens.sum()))
        print(f"HMM segmenter self loop bonus {self.self_loop_bonus_estimate} "
              f"num segments: {num_segments}, ratio {num_segments / float(x_lens.sum())}")
        return quantized, seg_idx


class VQBottleneckEMA(VQBottleneck):

    def __init__(self, in_dim, latent_dim, num_tokens,
                 decay=0.999, epsilon=1e-5,
                 avg_counts_after_reestim=True, **kwargs):
        super(VQBottleneckEMA, self).__init__(
            in_dim=in_dim, latent_dim=latent_dim, num_tokens=num_tokens, **kwargs)

        self.decay = decay
        self.epsilon = epsilon

        self.embedding.requires_grad = False
        self.embedding.weight.requires_grad = False
        self.register_buffer("ema_count", torch.zeros(num_tokens))
        self.register_buffer("ema_weight", self.embedding.weight.clone())
        self.avg_counts_after_reestim = avg_counts_after_reestim

    def codebook_train_hook(self, x, codes_one_hot):
        if self.training:
            # Sum one hots to 1 x num_tokens (save 1 for multiple codebooks in the future)
            # B x W x 1 x 1 x NTOKENS --> 1 x BW x NTOKENS
            one_hots = codes_one_hot.view(1, -1, codes_one_hot.size(-1)).detach()
            self.ema_count = (self.decay * self.ema_count
                              + (1 - self.decay) * one_hots.sum(dim=(0,1)))

            # n = num of codes
            n = torch.sum(self.ema_count, dim=0, keepdim=True)
            K = self.embedding.weight.size(0)
            ema_count = (self.ema_count + self.epsilon) / (n + K * self.epsilon) * n

            # N x W x 1 x C --> 1 x NW x C
            x_flat = x.view(-1, x.size(-1)).unsqueeze(0).detach()
            dw = torch.bmm(one_hots.transpose(1, 2), x_flat)
            dw = safe_squeeze(dw, 0)  # Remove multiple codebooks dimension
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding.weight.data = self.ema_weight / ema_count.unsqueeze(1)

    def post_reestim_hook(self):
        self.ema_weight = self.embedding.weight.clone()
        if self.avg_counts_after_reestim:
            self.ema_count.fill_(ema_count.mean())


class VQBottleneckSparse(VQBottleneck):
    def __init__(self, in_dim, latent_dim, num_tokens, dim=-1, commitment=0.25,
                 criterion_kwargs={}, use_copy_through=False):
        super(VQBottleneckSparse, self).__init__(in_dim, latent_dim, num_tokens,
                                                    dim=dim, commitment=commitment,
                                                    criterion='sparse',
                                                    criterion_kwargs=criterion_kwargs,
                                                    use_copy_through=use_copy_through)


class VQBottleneckSegmental(VQBottleneck):
    def __init__(self, in_dim, latent_dim, num_tokens, dim=-1, commitment=0.25,
                 criterion_kwargs={}, use_copy_through=False):
        super(VQBottleneckSegmental, self).__init__(in_dim, latent_dim, num_tokens,
                                                    dim=dim, commitment=commitment,
                                                    criterion='segmental',
                                                    criterion_kwargs=criterion_kwargs,
                                                    use_copy_through=use_copy_through)


class VQBottleneckNearest(VQBottleneck):
    def __init__(self, in_dim, latent_dim, num_tokens, dim=-1, commitment=0.25,
                 use_copy_through=False):
        super(VQBottleneckNearest, self).__init__(in_dim, latent_dim, num_tokens,
                                                  dim=dim, commitment=commitment,
                                                  criterion='nearest',
                                                  use_copy_through=use_copy_through)


class IndicesComputation(object):

    @staticmethod
    def nearest(inputs, codebook, temperature=None):
        with torch.no_grad():
            # inputs: NxD
            # codebook: KxD
            # NxK
            distances_matrix = torch.cdist(inputs, codebook)
            # Nx1
            if temperature is None:
                indices = torch.min(distances_matrix, dim=-1)[1].unsqueeze(1)
            else:
                probs = F.softmax(-distances_matrix / temperature, dim=-1)
                m = torch.distributions.Categorical(probs)
                indices = m.sample()
            return indices

    @staticmethod
    def segmental(inputs, codebook, segment_frac=0.1, segment_threshold=None):
        from distsup.modules.segment import calc
        distances_matrix = torch.cdist(inputs, codebook)
        indices, values = calc(distances_matrix, inputs.shape[0] * segment_frac, threshold=segment_threshold)
        indices = torch.from_numpy(indices).to(device=inputs.device).unsqueeze(1)
        return indices, values


class VectorQuantization(Function):

    @staticmethod
    def flatten(x):
        code_dim = x.size(-1)
        return x.view(-1, code_dim)

    @staticmethod
    def restore_shapes(codes, indices, target_shape):
        idx_shape = list(target_shape)
        idx_shape[-1] = 1
        return codes.view(*target_shape), indices.view(*idx_shape)

    @staticmethod
    def forward(ctx, inputs, codebook, commitment=0.25,
                criterion='nearest', criterion_kwargs={},
                use_copy_through=False):
        inputs_flat = VectorQuantization.flatten(inputs)
        compute_indices = getattr(IndicesComputation, criterion)
        indices = compute_indices(inputs_flat, codebook, **criterion_kwargs)
        if type(indices) is tuple:
            indices, values = indices
        codes = codebook[indices.view(-1), :]
        codes, indices = VectorQuantization.restore_shapes(
            codes, indices, inputs.shape)
        ctx.save_for_backward(codes, inputs, torch.FloatTensor([commitment]),
                              codebook, indices, torch.tensor([use_copy_through]))
        ctx.mark_non_differentiable(indices)
        if criterion != 'segmental':
            return codes, indices
        else:
            return codes, indices, torch.tensor(values)

    @staticmethod
    def backward(ctx, straight_through, unused_indices, unused_values=None):
        (codes, inputs, beta, codebook, indices, use_copy_through
         ) = ctx.saved_tensors

        # TODO: figure out proper vq loss reduction
        vq_loss = F.mse_loss(inputs, codes).detach()
        logger.log_scalar('vqlayer_loss', vq_loss)

        # gradient of vq_loss
        diff = 2 * (inputs - codes) / inputs.numel()

        commitment = beta.item() * diff

        if use_copy_through.item():
            code_disp = VectorQuantization.flatten(-diff + straight_through)
        else:
            code_disp = VectorQuantization.flatten(-diff)
        indices = VectorQuantization.flatten(indices)
        code_disp = (torch
                     .zeros_like(codebook)
                     .index_add_(0, indices.view(-1), code_disp))
        return straight_through + commitment, code_disp, None, None, None, None


quantize = VectorQuantization.apply


class SparseVectorQuantization(Function):
    @staticmethod
    def sparse(x, K):
        vec_len = torch.norm(x, dim=-1)
        _, idx = torch.sort(vec_len, descending=True)
        ret = torch.zeros_like(vec_len)
        ret[idx[:K]] = 1.0
        return ret, idx[K:], idx[:K]

    @staticmethod
    def restore_shapes(codes, indices, indices_weight, target_shape):
        idx_shape = list(target_shape)
        idx_shape[-1] = 1
        return codes.view(*target_shape), indices.view(*idx_shape), indices_weight.view(*idx_shape)

    @staticmethod
    def forward(ctx, inputs, codebook, commitment=0.25, criterion_kwargs={}):
        ratio = criterion_kwargs.pop('sparse_ratio', 0.2)
        assert len(criterion_kwargs) == 0, (
            f"unknown criterion_kwargs: {criterion_kwargs.keys()}")
        inputs_flat = VectorQuantization.flatten(inputs)
        indices = IndicesComputation.nearest(inputs_flat, codebook)
        K = int(inputs.numel() / inputs.size(-1) * ratio)
        indices_weight, removed_indices, remain_indices = SparseVectorQuantization.sparse(inputs_flat, K)

        codes = codebook[indices.view(-1), :] * indices_weight.unsqueeze(-1)
        codes, indices, indices_weight = SparseVectorQuantization.restore_shapes(
            codes, indices, indices_weight, inputs.shape)
        indices = indices * indices_weight.long() - (1 - indices_weight).long()
        ctx.save_for_backward(codes, inputs, torch.FloatTensor([commitment]),
                              codebook, indices, indices_weight, remain_indices)
        ctx.mark_non_differentiable(indices)
        ctx.mark_non_differentiable(indices_weight)
        return codes, indices

    @staticmethod
    def backward(ctx, straight_through, unused_indices):
        codes, inputs, beta, codebook, indices, indices_weight, remain_indices = ctx.saved_tensors

        # TODO: figure out proper vq loss reduction
        vq_loss = F.mse_loss(inputs, codes).detach()
        logger.log_scalar('vqlayer_loss', vq_loss)

        #diff = 2 * (inputs * indices_weight - codes) / indices_weight.sum() / inputs.size(-1)
        diff = 2 * (inputs - codes) / inputs.numel()

        commitment = beta.item() * diff

        code_disp = VectorQuantization.flatten(-diff)
        indices = indices.view(-1)[remain_indices]
        code_disp = code_disp[remain_indices]
        code_disp = (torch
                     .zeros_like(codebook)
                     .index_add_(0, indices, code_disp))
        #return straight_through * indices_weight + commitment, code_disp, None, None
        return straight_through + commitment, code_disp, None, None, None

sparse_quantize = SparseVectorQuantization.apply


class SOMBottleneck(nn.Module):
    def __init__(self, in_dim, latent_dim, num_tokens, dim=-1,
                 commitment=0.32, som_loss_mult=1.2,
                 prob_loss_mult=1.2,
                 smoothness_loss_mult=1.4):
        super(SOMBottleneck, self).__init__()
        assert isinstance(num_tokens, list)
        # num_tokens is a list in this case of the form: [5, 5]
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        # total number of embeddings (multiply elements of the token list)
        num_embeddings = np.prod(num_tokens)
        # The codebook
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        # transition probabilities in the discrete space
        # p(z_q_t | z_q_t-1)
        self.trans = nn.Linear(self.latent_dim, num_embeddings)
        # Linear projection from encoder output to embedding space
        self.projection = nn.Linear(in_dim, latent_dim)
        self.dim = dim
        self.commitment = commitment
        self.som_loss_mutl = som_loss_mult
        self.smoothness_loss_mult = smoothness_loss_mult
        self.prob_loss_mult = prob_loss_mult

    def loss_z_prob(self, z_q, z_dist_flat):
        """Computes the smoothness loss for the transitions given their probabilities."""
        # z_dist : bsz, time, ..
        # k: bsz, time
        # aggregate loss over all sequences
        bsz = z_q.size(0)
        time = z_q.size(1)
        z_q_old = torch.cat([z_q[:, 0:1], z_q[:, :-1]], dim=1)
        out_probabilities_old = F.softmax(self.trans(z_q_old), dim=-1).contiguous().view(bsz * time, -1)
        weighted_z_dist_prob = z_dist_flat * out_probabilities_old
        weighted_z_dist_prob = torch.mean(weighted_z_dist_prob)
        return weighted_z_dist_prob

    def loss_probabilities(self, z_q, k):
        # z_q: bsz*time, latent_dim
        z_q = z_q.contiguous().view(z_q.size(0), z_q.size(1), -1)
        z_q_old = torch.cat([z_q[:, 0:1], z_q[:, :-1]], dim=1)
        logits = self.trans(z_q_old)
        loss_p = F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)),
                                 k.contiguous().view(-1))
        return loss_p

    def z_q_ne(self, z_q, k, codebook):
        k_1 = k // self.num_tokens[1]
        k_2 = k % self.num_tokens[1]
        device = z_q.device.type
        batch_size = z_q.size(0)
        k1_not_top = torch.lt(k_1, self.num_tokens[0] - 1)
        k1_not_bottom = torch.gt(k_1, 0)
        k2_not_right = torch.lt(k_2, self.num_tokens[1] - 1)
        k2_not_left = torch.gt(k_2, 0)

        k1_up = torch.where(k1_not_top, torch.add(k_1, 1), k_1)
        k1_down = torch.where(k1_not_bottom, torch.sub(k_1, 1), k_1)
        k2_right = torch.where(k2_not_right, torch.add(k_2, 1), k_2)
        k2_left = torch.where(k2_not_left, torch.sub(k_2, 1), k_2)
        z_q_up = torch.where(k1_not_top.unsqueeze(-1), codebook[k1_up * self.num_tokens[1] + k_2],
                             torch.zeros(batch_size, self.latent_dim).to(device))
        z_q_down = torch.where(k1_not_bottom.unsqueeze(-1), codebook[k1_down * self.num_tokens[1] + k_2],
                               torch.zeros(batch_size, self.latent_dim).to(device))
        z_q_right = torch.where(k2_not_right.unsqueeze(-1), codebook[k_1 * self.num_tokens[1] + k2_right],
                                torch.zeros(batch_size, self.latent_dim).to(device))
        z_q_left = torch.where(k2_not_left.unsqueeze(-1), codebook[k_1 * self.num_tokens[1] + k2_left],
                               torch.zeros(batch_size, self.latent_dim).to(device))
        z_q_neighbors = torch.stack([z_q, z_q_up, z_q_down, z_q_right, z_q_left], dim=1)
        return z_q_neighbors

    def forward(self, x):
        inp_shape = x.shape[:-1]
        dim = self.dim
        needs_transpose = dim != -1 or dim != x.dim() - 1
        if needs_transpose:
            x = x.transpose(-1, dim).contiguous()
        # flatten x
        x = x.contiguous().view(-1, x.size(-1))
        # project to latent space
        z_e = self.projection(x)
        # compute distance between encodings and each of the embeddings
        # ze: N X D, codebook: K x D
        z_dist_flat = torch.cdist(z_e, self.embedding.weight)
        # z_dist_flat = torch.sum((z_e.unsqueeze(1) - self.embedding.weight.unsqueeze(0)).pow(2), dim=-1)
        # ---- Picks the index of the closest embedding for every encoding ----
        k = torch.argmin(z_dist_flat, dim=-1)
        # ---- Aggregates the respective closest embedding for every encoding ----
        z_q = self.embedding.weight[k]
        # ---- Get neighbours ----
        z_q_ne = self.z_q_ne(z_q, k, self.embedding.weight)
        if needs_transpose:
            z_q = z_q.transpose(-1, dim)
            z_q_ne = z_q_ne.transpose(-1, dim)

        k = k.contiguous().view(*inp_shape)

        loss = self.loss(z_e, z_q,
                         z_q_ne)  # compute loss
        z_q = z_q.contiguous().view(*inp_shape, -1)
        z_e = z_e.contiguous().view(*inp_shape, -1)
        # loss_z_prob = self.smoothness_loss_mult * self.loss_z_prob(z_q, z_dist_flat)
        # loss_transitions = self.prob_loss_mult * self.loss_probabilities(z_q, k)
        return z_q, loss, {"encodings": z_e,
                           "embeddings": z_q.detach(),
                           "indices": k.unsqueeze(-1)}

    def loss(self, z_e, z_q, z_q_ne):
        # update z_e (committing encodings to be close to embeddings)
        commit_loss = (z_e - z_q.detach()).pow(2).mean()
        # update z_q and its immediate neighbourhood
        som_loss = (z_e.unsqueeze(1).detach() - z_q_ne).pow(2).mean()
        loss = self.commitment * commit_loss + self.som_loss_mutl * som_loss
        return loss


class GaussMarkovBottleneck(nn.Module):
    def __init__(self, in_dim, latent_dim, transition_dim, dim=-1):
        super(GaussMarkovBottleneck, self).__init__()
        self.transition_fn = GatedTransition(latent_dim, transition_dim)
        self.combiner = Combiner(latent_dim, in_dim)
        self.z_q_0 = nn.Parameter(torch.zeros(latent_dim))
        self.dim = dim

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

    def sample_from_posterior(self, x):
        bsz, T_max, _ = x.size()
        z_prev = self.z_q_0.expand(bsz, self.z_q_0.size(0))
        mus = []
        scales = []
        zs = []
        for t in range(1, T_max + 1):
            z_mu, z_scale = self.combiner(z_prev, x[:, t - 1, :])
            # reparam sample
            z = distributions.Normal(z_mu, z_scale).rsample()
            mus.append(z_mu)
            scales.append(z_scale)
            zs.append(z)
            z_prev = z

        mus = torch.stack(mus, dim=1)
        scales = torch.stack(scales, dim=1)
        zs = torch.stack(zs, dim=1)

        return zs, (mus, scales)

    def kl_temporal(self, z_q, mean_q, scale_q, mask):
        mean_prior, scale_prior = self.transition(z_q)
        # batch mean of kl
        var_q = scale_q**2
        var_prior = scale_prior**2
        diff_mu = mean_prior - mean_q
        kl = 0.5 * (torch.log(var_prior) - torch.log(var_q) - 1. + var_q / var_prior + diff_mu ** 2 / var_prior)
        if mask is not None:
            return kl*mask.unsqueeze(-1)
        else:
            return kl

    def forward(self, x, mask=None, reduction="sum"):
        # x is the output of inference network
        # expects x to be of shape: bsz, time, dim
        assert len(x.shape) == 3
        dim = self.dim
        needs_transpose = dim != -1 or dim != x.dim() - 1
        if needs_transpose:
            x = x.transpose(-1, dim)
        z, z_variational_params = self.sample_from_posterior(x)
        latent_loss = self.kl_temporal(z, *z_variational_params, mask)
        # to make shapes compatible with the rest of the code
        # bsz, time, 1, latent_dim
        embedding = z_variational_params[0]
        if needs_transpose:
            z = z.transpose(-1, dim)
            latent_loss = latent_loss.transpose(-1, dim)
            embedding = embedding.transpose(-1, dim)
        # z needs to have four dimensions to be consistent with the rest of the code base
        z = z.unsqueeze(-2)
        # reduce the last two dimensions of the latent loss
        if reduction == "sum":
            latent_loss = latent_loss.sum((1, 2))
        else:
            latent_loss = latent_loss.mean((1, 2))
        return z, latent_loss, {"embeddings": embedding, "indices": None}


class SoftAttentionBottleneck(nn.Module):
    def __init__(self, in_dim, num_tokens, latent_dim=None, dim=-1,
                 initial_temp=1.0, min_temp=0.001, temp_factor_per_epoch=1.0,
                 straight_through_alpha=None, tie_weights=True):
        super(SoftAttentionBottleneck, self).__init__()
        self.projection = nn.Linear(in_dim, latent_dim)
        self.codebook = nn.Linear(latent_dim, num_tokens, bias=False)
        if tie_weights:
            self.attention = self.codebook
        else:
            self.attention = nn.Linear(latent_dim, num_tokens, bias=False)
        self.dim = dim
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.temp_factor_per_epoch = temp_factor_per_epoch
        self.straight_through_alpha = straight_through_alpha
        self.num_tokens = num_tokens

    def get_temperature(self):
        temp = self.initial_temp
        temp = temp * np.power(self.temp_factor_per_epoch, Globals.epoch - 1)
        return max(self.min_temp, temp)

    def forward(self, x, side_info=None, mask=None, enc_len=None):
        del side_info  # unused
        del mask  # unused
        del enc_len  # unused
        dim = self.dim
        needs_transpose = dim != -1 or dim != x.dim() - 1
        if needs_transpose:
            x = x.transpose(-1, dim).contiguous()
        x = self.projection(x)
        out = self.attention(x)
        scores = F.softmax(out / self.get_temperature(), dim=-1)
        indices = torch.argmax(scores, dim=-1, keepdim=True)
        codes = torch.matmul(scores, self.codebook.weight)
        if self.straight_through_alpha is not None:
            codes = codes + self.straight_through_alpha * (x - x.detach())
        if needs_transpose:
            codes = codes.transpose(-1, dim)
            indices = indices.transpose(-1, dim)
        return codes, None, {'indices': indices, 'embeddings': codes.detach()}


class GumbelBottleneck(SoftAttentionBottleneck):
    def __init__(self, in_dim, num_tokens, **kwargs):
        assert 'straight_through_alpha' not in kwargs
        super(GumbelBottleneck, self).__init__(
            in_dim=in_dim, num_tokens=num_tokens, **kwargs)

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, x, side_info=None, mask=None, enc_len=None):
        del side_info  # unused
        del mask  # unused
        del enc_len  # unused
        dim = self.dim
        needs_transpose = dim != -1 or dim != x.dim() - 1
        if needs_transpose:
            x = x.transpose(-1, dim).contiguous()
        x = self.projection(x)
        out = self.attention(x)

        if self.training:
            out = out + self.sample_gumbel(out.size()).to(out)

        scores = F.softmax(out / self.get_temperature(), dim=-1)
        indices = scores.argmax(dim=-1).detach()
        one_hot = F.one_hot(indices, num_classes=scores.size(-1))
        indices = indices.unsqueeze(-1)
        codes = scores + (one_hot.float() - scores).detach()
        codes = torch.matmul(codes, self.codebook.weight)

        if needs_transpose:
            codes = codes.transpose(-1, dim)
            indices = indices.transpose(-1, dim)
        return codes, None, {'indices': indices, 'embeddings': codes.detach()}


class UCBBottleneck(SoftAttentionBottleneck):
    def __init__(self, in_dim, num_tokens,
                 ucb_exploration=0.01, ucb_sequential_updates=True,
                 ucb_start_epoch=1, ucb_reset_every_epoch=True, **kwargs):
        assert 'straight_through_alpha' not in kwargs
        super(UCBBottleneck, self).__init__(
            in_dim=in_dim, num_tokens=num_tokens, **kwargs)
        self.ucb_exploration = ucb_exploration
        self.ucb_sequential_updates = ucb_sequential_updates
        self.ucb_start_epoch = ucb_start_epoch or 1
        self.ucb_reset_every_epoch = ucb_reset_every_epoch
        self.register_buffer(
            'counts', torch.ones(num_tokens, dtype=torch.float32))
        self.last_epoch = None

    def forward(self, x, side_info=None, mask=None, enc_len=None):
        del side_info  # unused
        del mask  # unused
        del enc_len  # unused

        is_new_epoch = (self.last_epoch != Globals.epoch)
        if (is_new_epoch and self.ucb_reset_every_epoch and self.training):
            self.counts.fill_(1.0)
            self.last_epoch = Globals.epoch

        dim = self.dim
        needs_transpose = dim != -1 or dim != x.dim() - 1
        if needs_transpose:
            x = x.transpose(-1, dim).contiguous()
        x = self.projection(x)
        out = self.attention(x)

        scores = F.softmax(out / self.get_temperature(), dim=-1)
        indices = scores.argmax(dim=-1).detach()

        if (self.training and self.ucb_exploration is not None and
                Globals.epoch >= self.ucb_start_epoch):
            scores, indices = self.apply_exploration(out, scores, indices)

        one_hot = F.one_hot(indices, num_classes=scores.size(-1))
        indices = indices.unsqueeze(-1)
        codes = scores + (one_hot.float() - scores).detach()
        codes = torch.matmul(codes, self.codebook.weight)

        if needs_transpose:
            codes = codes.transpose(-1, dim)
            indices = indices.transpose(-1, dim)
        return codes, None, {'indices': indices, 'embeddings': codes.detach()}

    def apply_exploration(self, out, scores, indices):
        explor = torch.zeros_like(scores)
        if self.ucb_sequential_updates:
            for b in range(scores.size(0)):
                # XXX This might be padded
                # TODO Cache counts here; currently it is slow
                for i in range(scores.size(1)):
                    explor[b, i, 0] = torch.sqrt(
                        self.ucb_exploration * 2.0 *
                        torch.log(self.counts.sum() / self.counts)).detach()
                    newout = (scores[b, i] + explor[b, i, 0]).detach().argmax()
                    self.counts[newout.item()] += 1

        else:  # UCB computed over batches - faster, but not accurate
            explor = torch.sqrt(
                self.ucb_exploration * 2.0 *
                torch.log(self.counts.sum() / self.counts))
            explor = explor.view(1, 1, -1)
            bc = torch.bincount(indices.view(-1).detach(),
                                minlength=self.counts.size(0))
            self.counts += bc.float() / bc.sum()

        scores = scores + explor.detach()
        new_indices = scores.argmax(dim=-1)
        logger.log_scalar('ucb/num_switches', (indices != new_indices).sum())
        return scores, new_indices


class MultipleBottlenecks(nn.Module):
    def __init__(self, in_dim, latent_dim, num_bottlenecks=10,
                 bottleneck=dict(
                     class_name=VQBottleneck,
                     num_tokens=16
                 )):
                 # bottleneck_latent_dim=64,):
        super(MultipleBottlenecks, self).__init__()
        self.num_bottlenecks = num_bottlenecks
        self.num_tokens = bottleneck['num_tokens']  # XXX For a single VQ
        self.bottlenecks = nn.ModuleDict({
            'bottleneck%d' % i: construct_from_kwargs(
                bottleneck, additional_parameters=dict(
                    in_dim=in_dim, latent_dim=latent_dim))
            for i in range(num_bottlenecks)})

    def forward(self, x, *args, **kwargs):
        # XXX side_info not yet supported
        nb = self.num_bottlenecks
        x = x.view(x.size(0), x.size(1), x.size(2), x.size(3) // nb, nb)
        all_outs = [
            self.bottlenecks['bottleneck%d' % i](x[:,:,:,:,i], *args, **kwargs)
            for i in range(nb)]

        ret = [torch.cat(outs, dim=-1) for outs in list(zip(*all_outs))[:2]]
        # XXX Drop other metrics ('embeddings', 'pre_bn_acts')
        ret.append({'indices':
            torch.cat([outs[2]['indices'] for outs in all_outs], dim=-1)})
        return ret
