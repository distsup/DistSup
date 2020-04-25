import random
import time
import numpy as np

from sklearn import cluster

import torch
import torch.nn as nn
import torch.nn.functional as F

from distsup.configuration import Globals
from distsup.logger import DefaultTensorLogger
from distsup.modules.bottlenecks import ReservoirSampler
from distsup.modules.encoders import Identity, Normalization

logger = DefaultTensorLogger()


class ManyCodebooksReservoirSampler(ReservoirSampler):
    def __init__(self, num_codebooks=1, num_samples=1024):
        super(ManyCodebooksReservoirSampler, self).__init__(num_samples)
        self.num_codebooks = num_codebooks

    def add(self, samples):
        assert samples.size(0) == self.num_codebooks
        self.ttot -= time.time()
        samples = samples.detach()
        if self.buffer is None:
            self.buffer = torch.empty(
                self.num_codebooks, self.n, samples.size(-1), device=samples.device)
        buffer = self.buffer
        if self.i < self.n:
            slots = self.n - self.i
            add_samples = samples[:, :slots]
            samples = samples[:, slots:]
            buffer[:, self.i: self.i + add_samples.size(1)] = add_samples
            self.i += add_samples.size(1)
            if samples.size(1) == 0:
                self.ttot += time.time()
                print(f"Res size {self.i}")
                return
        for cbk in range(self.num_codebooks):
            i = self.i
            for s in range(len(samples[cbk])):
                # warning, includes right end too.
                idx = random.randint(0, i)
                i += 1
                if idx < buffer.size(1):
                    buffer[cbk, idx] = samples[cbk, s]
        self.i = i
        self.ttot += time.time()

    def contents(self):
        return self.buffer[:, :self.i]


class ManyCodebooksVQBottleneck(nn.Module):
    """Vanilla VQ-VAE with three-term loss"""
    def __init__(self, num_codebooks, num_embeddings, embedding_dim,
                 codebook_cost=1.0,
                 commitment_cost=0.25,
                 reestimation_reservoir_size=None,
                 reestimate_every_epochs=0,
                 reestimate_every_iters=0,
                 reestimate_every_iters_expansion=0,
                 reestimate_max_iters=0,
                 reestimate_max_epochs=0,
                 bottleneck_enforce_from_epoch=-1,
                 log_input_norms=False,
                 normalization='none',
                 normalization_nary=2,
                 normalization_set_affine=None):
        super(ManyCodebooksVQBottleneck, self).__init__()
        self.num_codebooks = num_codebooks
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.codebook_cost = codebook_cost
        self.commitment_cost = commitment_cost

        self.batch_norm = Normalization(normalization, normalization_nary,
            embedding_dim, set_affine=normalization_set_affine)
        self.log_input_norms = log_input_norms

        embedding = torch.zeros(num_codebooks, num_embeddings, embedding_dim)
        self.embedding = nn.Parameter(embedding)
        embedding.uniform_(-1/num_embeddings, 1/num_embeddings)

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
            self.reestimation_reservoir = ManyCodebooksReservoirSampler(
                num_codebooks, reestimation_reservoir_size)
            self.reestimate_every_epochs = reestimate_every_epochs
            self.reestimate_last_epoch = Globals.epoch
            self.reestimate_every_iters = reestimate_every_iters
            self.reestimate_every_iters_expansion = reestimate_every_iters_expansion
            self.reestimate_max_epochs = reestimate_max_epochs
            self.reestimate_max_iters = reestimate_max_iters
            assert reestimate_every_epochs or reestimate_every_iters
        else:
            self.reestimation_reservoir = None
        self.skip_quant_message = -1
        self.last_msg = -1

    def codebook_train_hook(self, x_flat, encodings, M):
        pass

    def post_reestim_hook(self):
        pass

    # Based on https://github.com/bshall/VectorQuantizedVAE
    def forward(self, x):
        if self.reestimation_reservoir and self.training:
            self.reestimate()

        B, C, H, W = x.size()
        N, M, D = self.embedding.size()
        assert C == N * D, f"C={C}, N={N}, D={D}"

        x = x.view(B, N, D, H, W).contiguous()

        if self.log_input_norms:
            norms = torch.norm(x.permute(0, 1, 3, 4, 2).contiguous().view(-1, D), dim=1)
            logger.log_scalar('vq_input_norms_pre_bn/mean', torch.mean(norms))
            logger.log_scalar('vq_input_norms_pre_bn/std', torch.std(norms))

        if self.batch_norm is not Identity:
            x = x.view(B * N, D, H, W)
            x = self.batch_norm(x)
            x = x.view(B, N, D, H, W)

        x = x.permute(1, 0, 3, 4, 2).contiguous()

        if self.log_input_norms:
            norms = torch.norm(x.view(-1, D), dim=1)
            logger.log_scalar('vq_input_norms_post_bn/mean', torch.mean(norms))
            logger.log_scalar('vq_input_norms_post_bn/std', torch.std(norms))

        x_flat = x.detach().reshape(N, -1, D)

        norms = torch.norm(x_flat, dim=-1)
        stats = {'enc_mean': torch.mean(norms), 'enc_std': torch.std(norms)}

        if self.training and self.reestimation_reservoir:
            self.reestimation_reservoir.add(x_flat)

        if Globals.epoch < self.bottleneck_enforce_from_epoch:
            if self.skip_quant_message < Globals.epoch:
                print(f"Skipping quantization in epoch {Globals.epoch}")
                self.skip_quant_message = Globals.epoch
            out = x.permute(1, 0, 4, 2, 3).contiguous().view(B, C, H, W)
            stats['pplx'] = 0
            stats['pplx_fixed'] = 0
            return out, 0, stats
        else:
            distances = torch.baddbmm(torch.sum(self.embedding ** 2, dim=2).unsqueeze(1) +
                                      torch.sum(x_flat ** 2, dim=2, keepdim=True),
                                      x_flat, self.embedding.transpose(1, 2),
                                      alpha=-2.0, beta=1.0)

            indices = torch.argmin(distances, dim=-1)
            encodings = F.one_hot(indices, M).float()
            quantized = torch.gather(self.embedding, 1,
                                     indices.unsqueeze(-1).expand(-1, -1, D))
            quantized = quantized.view_as(x)

            self.codebook_train_hook(x_flat, encodings, M)

            vq_loss = F.mse_loss(x.detach(), quantized) * self.codebook_cost
            vq_loss += F.mse_loss(x, quantized.detach()) * self.commitment_cost

            quantized = x + (quantized - x).detach()

            avg_probs = torch.mean(encodings, dim=1)
            stats['pplx'] = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1)).sum()
            stats['pplx_fixed'] = (-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1)).sum()

            out = quantized.permute(1, 0, 4, 2, 3).reshape(B, C, H, W)
            return out, vq_loss, stats

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

        num_codebooks = self.embedding.size(0)
        num_clusters = self.embedding.size(1)

        encodings = self.reestimation_reservoir.contents()
        if encodings.size(1) < num_clusters:
            print(f"Skipping reestimation, too few samples")
            return
        encodings = encodings.cpu().numpy()

        for i in range(num_codebooks):
            clustered, *_ = cluster.k_means(encodings[i], num_clusters)
            self.embedding.data[i] = (torch.tensor(clustered).to(self.embedding.device))

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

        # Updates EMA
        self.post_reestim_hook()


class ManyCodebooksEMABottleneck(ManyCodebooksVQBottleneck):
    def __init__(self, num_codebooks, num_embeddings, embedding_dim,
                 decay=0.999, epsilon=1e-5,
                 ema_reestim_zero_counts=False, **kwargs):
        super(ManyCodebooksEMABottleneck, self).__init__(
            num_codebooks=num_codebooks, num_embeddings=num_embeddings,
            embedding_dim=embedding_dim, **kwargs)
        self.decay = decay
        self.epsilon = epsilon

        self.embedding.requires_grad = False
        self.register_buffer("ema_count", torch.zeros(num_codebooks,
                                                      num_embeddings))
        self.register_buffer("ema_weight", self.embedding.data.clone())
        self.ema_reestim_zero_counts = ema_reestim_zero_counts

    def codebook_train_hook(self, x_flat, encodings, M):
        if self.training:
            self.ema_count = (self.decay * self.ema_count +
                              (1 - self.decay) * torch.sum(encodings, dim=1))

            n = torch.sum(self.ema_count, dim=-1, keepdim=True)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.bmm(encodings.transpose(1, 2), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding.data = self.ema_weight / self.ema_count.unsqueeze(-1)

    def post_reestim_hook(self):
        print('Setting EMA weight')
        self.ema_weight = self.embedding.data.clone()
        if self.ema_reestim_zero_counts:
            self.ema_count.zero_()

