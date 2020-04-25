import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from distsup.utils import construct_from_kwargs
from distsup.models import base


class Residual(nn.Module):
    def __init__(self, in_ch, out_ch=None, ksp1=(3,1,1), ksp2=(1,1,0),
                 batch_norm=False):
        """
            ksp (tuple): kernel size, stride, padding
        """
        super(Residual, self).__init__()
        out_ch = out_ch or in_ch
        layers = [
            nn.ReLU(True),
            nn.Conv2d(in_ch, out_ch, *ksp1, bias=False),
            nn.BatchNorm2d(out_ch) if batch_norm else None,
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, *ksp2, bias=False),
            nn.BatchNorm2d(out_ch) if batch_norm else None]
        self.block = nn.Sequential(*[l for l in layers if l is not None])
        if out_ch != in_ch:
            stride = ksp1[1]
            self.short = nn.Conv2d(in_ch, out_ch, 1, stride, 0)

    def forward(self, x):
        return self.block(x) + (self.short(x) if hasattr(self, 'short') else x)


class ConvEncoder(nn.Module):
    def __init__(self, channels, latent_dim, embedding_dim, batch_norm=False):
        super(ConvEncoder, self).__init__()
        layers = [
            nn.Conv2d(3, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels) if batch_norm else None,
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels) if batch_norm else None,
            Residual(channels, batch_norm=batch_norm),
            Residual(channels, batch_norm=batch_norm),
            nn.Conv2d(channels, latent_dim * embedding_dim, 1)]
        self.encoder = nn.Sequential(*[l for l in layers if l is not None])

    def forward(self, x):
        return self.encoder(x)


class ConvDecoder(nn.Module):
    def __init__(self, channels, latent_dim, embedding_dim, batch_norm=False):
        super(ConvDecoder, self).__init__()
        layers = [
            nn.Conv2d(latent_dim * embedding_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels) if batch_norm else None,
            Residual(channels, batch_norm=batch_norm),
            Residual(channels, batch_norm=batch_norm),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels) if batch_norm else None,
            nn.ReLU(True),
            nn.ConvTranspose2d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels) if batch_norm else None,
            nn.ReLU(True),
            nn.Conv2d(channels, 3 * 256, 1)]
        self.decoder = nn.Sequential(*[l for l in layers if l is not None])

    def forward(self, x):
        x = self.decoder(x)
        B, _, H, W = x.size()
        x = x.view(B, 3, 256, H, W).permute(0, 1, 3, 4, 2)
        dist = Categorical(logits=x)
        return dist


class FeedForwardLearner(base.Model):
    def __init__(self,
                 bottleneck=dict(
                    class_name='ManyCodebooksVQBottleneck',
                    num_codebooks=8,
                    num_embeddings=128,
                    embedding_dim=32),
                encoder=dict(
                    class_name='ConvEncoder',
                    channels=256,
                    batch_norm=True),
                decoder=dict(
                    class_name='ConvDecoder',
                    channels=256,
                    batch_norm=True),
                 **kwargs):
        super(FeedForwardLearner, self).__init__(**kwargs)

        self.bottleneck = construct_from_kwargs(bottleneck)
        kw = {'latent_dim': self.bottleneck.num_codebooks,
              'embedding_dim': self.bottleneck.embedding_dim}
        self.encoder = construct_from_kwargs(encoder, additional_parameters=kw)
        self.decoder = construct_from_kwargs(decoder, additional_parameters=kw)
        self.N = None

    def minibatch_loss(self, batch):
        images = batch['features']
        targets = ((images + 0.5) * 255).long()

        N = np.prod(images.size()[1:])
        if self.N is None:
            self.N = N
        else:
            assert N == self.N

        x = self.encoder(images)
        x, vq_loss, stats = self.bottleneck(x)
        dist = self.decoder(x)

        num_codebooks, num_tokens = self.bottleneck.embedding.size()[:2]
        KL = num_codebooks * 8 * 8 * np.log(num_tokens)
        logp = dist.log_prob(targets).sum((1, 2, 3)).mean()
        stats['logp'] = logp
        stats['loss'] = - logp / N + vq_loss
        stats['elbo'] = (KL - logp) / N
        stats['bpd'] = stats['elbo'] / np.log(2)
        # stats['distrib'] = dist
        return stats['loss'], stats

    # XXX Unused ?
    def parameters(self, with_codebook=True):
        if with_codebook:
            return super(FeedForwardLearner, self).parameters()
        else:
            return itertools.chain(self.encoder.parameters(),
                                   self.decoder.parameters())


class ResNetN(nn.Module):
    def __init__(self, args, channels, latent_dim, num_embeddings,
                 embedding_dim, bottleneck='VQVAE',
                 batch_norm=False, n=12, num_classes=10):
        super(ResNetN, self).__init__()
        assert n % 6 == 0
        l = [nn.Conv2d(3, 64, 3, 1, 1)]
        if batch_norm:
            l += [nn.BatchNorm2d(64)]
        self.conv = nn.Sequential(*l)
        self.block1 = nn.Sequential(
            *[Residual(64, 64, (3, 1, 1), batch_norm=batch_norm)
              for i in range(n // 6)])
        self.block2 = nn.Sequential(
            *[Residual(64, 128, (3, 2, 1), (3, 1, 1), batch_norm=batch_norm)] + [
              Residual(128, 128, (3, 1, 1), batch_norm=batch_norm)
              for i in range(n // 6 -1)])
        self.block3 = nn.Sequential(
            *[Residual(128, 256, (3, 2, 1), (3, 1, 1), batch_norm=batch_norm)] + [
              Residual(256, 256, (3, 1, 1), batch_norm=batch_norm)
              for i in range(n // 6 - 2)] + [
              Residual(256, 256, (3, 1, 1), batch_norm=batch_norm)])
        self.fc = nn.Linear(256, num_classes)

        self.bottleneck_after_block = args.bottleneck_after_block
        if bottleneck is None or bottleneck.lower() == "none":
            self.codebook = None
        elif bottleneck == 'VQVAE':
            self.codebook = VQEmbedding(
                latent_dim, num_embeddings, embedding_dim,
                codebook_cost=args.codebook_cost,
                reestimation_reservoir_size=args.reestimation_reservoir_size,
                reestimate_every_epochs=args.reestimate_every_epochs,
                reestimate_every_iters=args.reestimate_every_iters,
                reestimate_every_iters_expansion=args.reestimate_every_iters_expansion,
                reestimate_max_iters=args.reestimate_max_iters,
                reestimate_max_epochs=args.reestimate_max_epochs,
                bottleneck_enforce_from_epoch=args.bottleneck_enforce_from_epoch,)
        elif bottleneck == 'EMA':
            self.codebook = VQEmbeddingEMA(
                latent_dim, num_embeddings, embedding_dim, decay=args.ema_decay)
        else:
            raise ValueError

    def forward(self, x, labels, N):
        stats = {'enc_mean': 0, 'enc_std': 0}  # Dummy
        x = self.conv(x)
        for idx in range(1, 4):
            x = getattr(self, f'block{idx}')(x)
            if self.bottleneck_after_block == idx and self.codebook is not None:
                x, vq_loss, stats = self.codebook(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        stats['loss'] = F.cross_entropy(x, labels, reduction='mean')
        stats['acc'] = torch.mean((labels == torch.argmax(x, dim=1)).float()) * 100.0
        if self.codebook is not None:
            stats['loss'] += vq_loss
        return None, stats['loss'], stats

    def parameters(self, with_codebook=True):
        if with_codebook:
            return super(ResNetN, self).parameters()
        else:
            return itertools.chain(self.conv.parameters(),
                                   self.block1.parameters(),
                                   self.block2.parameters(),
                                   self.block3.parameters(),
                                   self.fc.parameters())
