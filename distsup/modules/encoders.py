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

from collections import OrderedDict

import sys

import numpy as np

import torch
from torch import nn
import numpy as np
from distsup.modules.recurrent.encoders import Encoder
import distsup.utils as utils

from distsup.modules.recurrent import encoders as rec_encs


def zero_nan(tensor_, val=0.):
    for x in tensor_:
        x[x != x] = val   # replace all nan/inf in gradients to zero


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        del args  # unused
        del kwargs  # unused
        super(Identity, self).__init__()
        self.length_reduction = 1

    def forward(self, x, features_lens=None):
        if features_lens is not None:
            return (x, features_lens)
        else:
            return x


class Normalization(nn.Module):
    def __init__(self, norm_type, nary, input_size, set_affine=None):
        """
        Args
            set_affine (None or bool): Override the PyTorch default (True for
                BatchNorm, False for InstanceNorm)
        """
        super(Normalization, self).__init__()
        self.nary = nary
        kw = {'affine': set_affine} if set_affine is not None else {}
        if norm_type == 'batch_norm':
            if nary == 1:
                self.batch_norm = nn.BatchNorm1d(input_size, **kw)
            elif nary == 2:
                self.batch_norm = nn.BatchNorm2d(input_size, **kw)
            else:
                raise ValueError(
                    "Unknown nary for {} normalization".format(norm_type))
        elif norm_type == 'instance_norm':
            if nary == 1:
                self.batch_norm = nn.InstanceNorm1d(input_size, **kw)
            elif nary == 2:
                self.batch_norm = nn.InstanceNorm2d(input_size, **kw)
            else:
                raise ValueError(
                    "Unknown nary for {} normalization".format(norm_type))
        elif not norm_type or norm_type == 'none':
            self.batch_norm = Identity()
        else:
            raise ValueError(
                """Unknown normalization type {}.
                   Possible are: batch_norm, instance_norm or none"""
                .format(norm_type))

    def forward(self, x):
        # Remove any NaNs from higher layers
        # if isinstance(x, torch.nn.utils.rnn.PackedSequence):
        #     zero_nan(x.data)
        # else:
        #     zero_nan(x)

        if self.nary >= 2:
            y = self.batch_norm(x)
            return y
        elif self.nary == 1:
            if isinstance(x, torch.nn.utils.rnn.PackedSequence):
                y = self.batch_norm(x.data)
                return torch.nn.utils.rnn.PackedSequence(y, x.batch_sizes)
            else:
                y = self.batch_norm(x)
                return y


class BatchRNN(nn.Module):
    """
    RNN with normalization applied between layers.
    :param input_size: Size of the input
    :param hidden_size: Size of hidden state
    :param rnn_type: Class for initializing RNN
    :param bidirectional: is it bidirectional
    :param normalization: String, what type of normalization to use.
                          Possible options: 'batch_norm', 'instance_norm'
    """
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM,
                 bidirectional=False, normalization=None,
                 projection_size=0, residual=False, subsample=False, bias=False):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.residual = residual
        self.batch_norm = Normalization(normalization, 1, input_size)

        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=bias)
        self.num_directions = 2 if bidirectional else 1

        self.subsample = subsample

        if projection_size > 0:
            self.projection = torch.nn.Linear(
                hidden_size * self.num_directions, projection_size,
                bias=False)
        else:
            self.projection = None

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    """
    :input x: input of size t x bs x f
    """
    def forward(self, x):
        if self.residual:
            res = x
        x = self.batch_norm(x)
        x, _ = self.rnn(x)
        if self.subsample:
            x, lengths = nn.utils.rnn.pad_packed_sequence(x)
            x = x[::2]
            x = nn.utils.rnn.pack_padded_sequence(x, (lengths + 1) // 2)
        if self.projection:
            if isinstance(x, torch.nn.utils.rnn.PackedSequence):
                x_data = self.projection(x.data)
                x = torch.nn.utils.rnn.PackedSequence(x_data, x.batch_sizes)
            else:
                x = self.projection(x)
        elif self.bidirectional:
            # (TxBSxH*2) -> (TxBSxH) by sum
            if isinstance(x, torch.nn.utils.rnn.PackedSequence):
                x_data = (x.data.view(x.data.size(0), 2, -1)
                           .sum(1).view(x.data.size(0), -1))
                x = torch.nn.utils.rnn.PackedSequence(x_data, x.batch_sizes)
            else:
                x = (x.view(x.size(0), x.size(1), 2, -1)
                      .sum(2).view(x.size(0), x.size(1), -1))
        if self.residual:
            if self.subsample:
                res, lengths = nn.utils.rnn.pad_packed_sequence(res)
                res = res[::2]
                res = nn.utils.rnn.pack_padded_sequence(res, (lengths + 1) // 2)

            x_data = torch.nn.functional.relu(x.data + res.data)
            x = torch.nn.utils.rnn.PackedSequence(x_data, x.batch_sizes)
        return x


class SequentialWithOptionalAttributes(nn.Sequential):
    def forward(self, input, *args):
        for module in self._modules.values():
            params_count = module.forward.__code__.co_argcount
            # params_count is self + input + ...
            input = module(input, *args[:(params_count-2)])
        return input


##############################################################################
# Deep speech implementation based on
# https://github.com/SeanNaren/deepspeech.pytorch
##############################################################################

def makeRnn(rnn_input_size, rnn_hidden_size, rnn_nb_layers, rnn_projection_size,
            rnn_type, rnn_dropout, rnn_residual, normalization,
            rnn_subsample, rnn_bidirectional, rnn_bias):
    if rnn_subsample is None:
        rnn_subsample = []
    if rnn_dropout > 0.0:
        rnn_dropout = nn.modules.Dropout(p=rnn_dropout)
    else:
        rnn_dropout = None

    rnns = []
    rnn = BatchRNN(input_size=rnn_input_size,
                   hidden_size=rnn_hidden_size,
                   rnn_type=rnn_type, bidirectional=rnn_bidirectional,
                   normalization=None,
                   projection_size=rnn_projection_size,
                   subsample=(0 in rnn_subsample),
                   bias=rnn_bias)
    rnns.append(('0', rnn))
    rnn_cumulative_stride = 2**(0 in rnn_subsample)
    for i in range(rnn_nb_layers - 1):
        rnn = BatchRNN(input_size=(rnn_hidden_size
                                   if rnn_projection_size == 0
                                   else rnn_projection_size),
                       projection_size=rnn_projection_size,
                       hidden_size=rnn_hidden_size,
                       rnn_type=rnn_type,
                       bidirectional=rnn_bidirectional,
                       normalization=normalization,
                       residual=rnn_residual,
                       subsample=(i+1 in rnn_subsample),
                       bias=rnn_bias)
        rnn_cumulative_stride *= 2**(i + 1 in rnn_subsample)
        if rnn_dropout:
            rnns.append(('{}_dropout'.format(i+1), rnn_dropout))
        rnns.append(('{}'.format(i + 1), rnn))
    rnns = SequentialWithOptionalAttributes(OrderedDict(rnns))

    return rnns, rnn_cumulative_stride


class RNNStack(nn.Module):
    def __init__(self,
                 in_channels,
                 image_height=1,
                 hid_channels=64,
                 nb_layers=1,
                 projection_channels=None,
                 type='LSTM',
                 dropout=0,
                 residual=True,
                 normalization='none',
                 subsample=(),
                 bidirectional=True,
                 bias=True,
                 preserve_len=False,
                 **kwargs):
        super(RNNStack, self).__init__(**kwargs)
        self.rnns, self.length_reduction = makeRnn(
            rnn_input_size=in_channels * image_height,
            rnn_hidden_size=hid_channels,
            rnn_nb_layers=nb_layers,
            rnn_projection_size=projection_channels or hid_channels,
            rnn_type=getattr(nn, type),
            rnn_dropout=dropout,
            rnn_residual=residual,
            normalization=normalization,
            rnn_subsample=subsample,
            rnn_bidirectional=bidirectional,
            rnn_bias=bias)
        self.preserve_len = preserve_len

    def forward(self, x, lens=None):
        x_in = x
        N, W, H, C = x.size()
        x = x.view(N, W, H * C)
        ret_lens = True
        if lens is None:
            ret_lens = False
            lens = torch.empty((N,), dtype=torch.int64).fill_(W)
        x = nn.utils.rnn.pack_padded_sequence(
            x, lens, batch_first=True, enforce_sorted=True)
        x = self.rnns(x)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.unsqueeze(2)
        if self.preserve_len and x_in.shape[1] != x.shape[1]:
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, x_in.shape[1] - x.shape[1]))
        if ret_lens:
            return x, lens
        else:
            return x


class DownsamplingEncoder(nn.Module):
    def __init__(self, length_reduction=1, image_height=None, in_channels=None):
        super(DownsamplingEncoder, self).__init__()
        del image_height
        del in_channels
        self.length_reduction = length_reduction

    def forward(self, features, features_lens=None):
        features = features[:, ::self.length_reduction]

        return (features, features_lens // self.length_reduction) \
            if features_lens is not None else features


class DeepSpeech2(nn.Module):
    """
    Deep Speech 2 implementation
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 input_height=None,
                 image_height=None,
                 num_input_channels=None,
                 in_channels=None,
                 conv_normalization='batch_norm',
                 conv_strides=[[1, 2], [3, 1]],
                 conv_kernel_sizes=[[7, 7], [7, 7]],
                 conv_num_features=[32, 32],
                 conv_nonlinearity='hardtanh',
                 rnn_hidden_size=768,
                 rnn_nb_layers=5,
                 rnn_projection_size=0,
                 rnn_type=nn.LSTM,
                 rnn_dropout=0.0,
                 rnn_residual=False,
                 rnn_normalization='batch_norm',
                 rnn_subsample=None,
                 rnn_bidirectional=True,
                 rnn_bias=False,
                 **kwargs):
        super(DeepSpeech2, self).__init__(**kwargs)

        if (num_input_channels == None) + (in_channels == None) != 1:
            raise Exception("The DeepSpeechEncoder should be provided with "
                            "exactly 1 of num_input_channels or in_channels.")
        if num_input_channels is None:
            num_input_channels = in_channels
            assert num_input_channels is not None

        if (input_height == None) + (image_height == None) != 1:
            raise Exception("The DeepSpeechEncoder should be provided with "
                            "exactly 1 of input_height or image_height.")
        if input_height is None:
            input_height = image_height
            assert input_height is not None

        self.makeConv(num_input_channels, conv_strides, conv_kernel_sizes,
                      conv_num_features, conv_normalization, conv_nonlinearity)

        # Compute output size of self.conv
        # Warning: forwarding zeros in train() mode will set BN params to NaN!
        self.eval()
        features = torch.ones((1, num_input_channels, 100, input_height))
        after_conv_size = self.conv.forward(features).size()
        self.rnn_input_size = self.computeRnnInputSize(after_conv_size)

        self.rnn_nb_layers = rnn_nb_layers

        if self.rnn_nb_layers > 0:
            self.rnns, self.rnn_cumulative_stride = makeRnn(
                self.rnn_input_size, rnn_hidden_size,
                rnn_nb_layers, rnn_projection_size,
                rnn_type, rnn_dropout, rnn_residual, rnn_normalization,
                rnn_subsample, rnn_bidirectional, rnn_bias)
            self.output_dim = rnn_hidden_size
        else:
            self.rnn_cumulative_stride = 1
            self.output_dim = self.rnn_input_size
        self.train()

    @property
    def length_reduction(self):
        return int(self.conv_cumative_stride * self.rnn_cumulative_stride)

    def makeConv(self, num_channels, conv_strides, conv_kernel_sizes,
                 conv_num_features, normalization, nonlinearity='hardtanh'):
        assert (len(conv_strides) == len(conv_kernel_sizes)
                == len(conv_num_features))

        conv_cum_stride = 1.0
        for i in range(len(conv_strides)):
            conv_cum_stride *= conv_strides[i][0]

        self.conv_cumative_stride = conv_cum_stride
        self.conv_activation = {
          'tanh': nn.Tanh(),
          'relu': nn.ReLU(inplace=True),
          'hardtanh' : nn.Hardtanh(0, 20, inplace=True),
          'leakyrelu' : nn.LeakyReLU(0.1, inplace=True)
        }[nonlinearity]

        layers = []
        for nf, ks, cs, in zip(conv_num_features,
                               conv_kernel_sizes,
                               conv_strides):
            layers.extend([
                nn.Conv2d(num_channels, nf,
                          kernel_size=ks, stride=cs,
                          padding=((ks[0] - 1) // 2, (ks[1] - 1) // 2)),
                Normalization(normalization, 2, nf),
                self.conv_activation
                ])
            num_channels = nf
        self.conv = nn.Sequential(*layers)

        # Post convolution layer with the convention data layout (allows hooking probes and heads)
        self.post_conv = Identity()

    def computeRnnInputSize(self, after_conv_size):
        rnn_input_size = after_conv_size[1] * after_conv_size[3]
        return rnn_input_size

    def forward(self, features, features_lens=None):
        if features_lens is None:
            return_features_lens = False
            features_lens = torch.empty(
                features.size(0), device=features.device,
                dtype=torch.int64).fill_(features.size(1))
        else:
            return_features_lens = True

        # bs x t x f x c -> bs x c x t x f
        features = features.permute(0, 3, 1, 2)
        features = self.conv(features)
        (batch_size, unused_num_channels, num_timestp, unused_num_features
         ) = features.size()

        # bs x c x t x f -> bs x t x f x c
        features = features.permute(0, 2, 3, 1)
        features = self.post_conv(features)

        features_lens = ((
            features_lens + self.conv_cumative_stride - 1
            ) / self.conv_cumative_stride).int()

        if self.rnn_nb_layers > 0:
            # bs x t x f x c -> t x bs x c x f - > t x bs x (c x f)
            features = features.permute(1, 0, 3, 2).contiguous()
            features = features.view(num_timestp, batch_size, -1).contiguous()

            # The input is already permuted for RNN
            assert features_lens[0] <= features.size()[0]
            features, features_lens = self.forwardRnn(features, features_lens)
            # The input is in canonical form (size[1] is time)
            assert features_lens[0] == features.size()[1]

        return (features, features_lens) if return_features_lens else features

    def forwardRnn(self, features, features_lens):
        # Inputs (t x bs x d)
        features = nn.utils.rnn.pack_padded_sequence(
            features, features_lens.data.cpu().numpy())

        features = self.rnns(features)

        # Outputs (bs x t x d)
        features, features_lens = nn.utils.rnn.pad_packed_sequence(
            features, batch_first=True)

        # Outputs (bs x t x 1 x d)
        return features.unsqueeze(2), features_lens


class GlobalEncoder(torch.nn.Module):
    def __init__(self, embedding_dim, in_channels=1):
        d = embedding_dim
        super(GlobalEncoder, self).__init__()
        self.encoder = torch.nn.Sequential( #28/32
            torch.nn.Conv2d(in_channels, d, kernel_size=3, stride=2, bias=False), # 13/15
            torch.nn.BatchNorm2d(d),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(d, d, kernel_size=3, stride=2, bias=False), # 6/7
            torch.nn.BatchNorm2d(d),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(d, d, kernel_size=3, stride=2, bias=True), # 2/3
            #torch.nn.Conv2d(d, d // 2, kernel_size=3, stride=2, bias=True), # 2/3
        )
        #self.apply(weights_init)

    def forward(self, batch):
        x=batch['features'].permute(0, 3, 2, 1)
        #x: (B, 1, D, T)
        x = self.encoder(x) #(B, d, 2/3, T')
        x = torch.mean(x, dim=(2,3))
        #x = torch.cat((torch.mean(x, dim=(2,3)), torch.std(x, dim=(2,3)) + 1e-20), dim=-1)
        return x


class AntiCausalRNNEncoder(nn.Module):
    # anti-causal encoder that summarizes information from the future time steps
    # of a time sequence
    def __init__(self,
                 input_size, num_layers, hidden_size, projection_size,
                 subsample, dropout=0., rnn_type="lstmp", in_channel=1):
        super(AntiCausalRNNEncoder, self).__init__()
        assert isinstance(subsample, list)
        self.length_reduction = np.cumprod(subsample)
        self.encoder = rec_encs.Encoder(
            rnn_type, input_size, num_layers, hidden_size, projection_size,
            subsample, dropout, in_channel)

    def forward(self, x, x_lens):
        # x shape: bsz, w, h, c
        # or: bsz, time, freq_bins, channels
        bsz, w, h, c = x.size()
        x = x.contiguous().view(bsz, w, h*c)
        # reverse the time sequence
        x_rev = utils.reverse_sequences(x, x_lens)
        out, out_lens, _ = self.encoder(x_rev, x_lens)
        out = utils.reverse_sequences(out, out_lens)
        out_mask = utils.get_mini_batch_mask(out, out_lens).to(x.device)
        return out.unsqueeze(-2), out_mask


class OneHot(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=None):
        super(OneHot, self).__init__()
        # For compatibility with the nn.Embedding.
        if embedding_dim is not None:
            assert num_embeddings == embedding_dim
        self.num_embeddings = num_embeddings

    def forward(self, x):
        ret_shape = x.size() + (self.num_embeddings,)
        ret = torch.zeros(ret_shape, device=x.device)
        ret.scatter_(-1, x.unsqueeze(-1), 1)
        return ret
