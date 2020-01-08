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

import argparse
import errno
import functools
import importlib
import json
import os
import re
import six
import json
import math
import sys


import numpy as np
import torch
import torch.nn as nn
import logging

import editdistance

log_msg_cache = set()


class Reshape(nn.Module):
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.contiguous().view(x.size(0), *self.shape)


class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims).contiguous()


def is_scalar(t):
    return isinstance(t, (float, int)) or (torch.is_tensor(t) and len(t.size()) == 0)


def maybe_get_scalar(t):
    if isinstance(t, (float, int)):
        return True, t
    if torch.is_tensor(t) and len(t.size()) == 0:
        return True, t.item()
    return False, None


def safe_squeeze(t, dim):
    assert t.size(dim) == 1
    return t.squeeze(dim)


class DebugStats:
    def __init__(self, logger):
        self.reset()
        self.logger = logger

    def reset(self):
        self.acts = []
        self.bw_acts = []

    @staticmethod
    def make_range(values):
        m = min(values)
        M = max(values)
        r = M - m
        return (m - 0.1 * r, M + 0.1 * r)

    def _show(self, what):
        import matplotlib.pyplot as plt

        xs = [x[1] for x in what]
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.title.set_text("mean")
        ax.scatter(x=xs, y=[x[0] for x in what])
        ax.set_xlim(DebugStats.make_range(xs))

        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() +
                        ax.get_yticklabels()):
            item.set_fontsize(8)

        xs = [x[2] for x in what]
        ax = fig.add_subplot(122)
        ax.title.set_text("variance")
        ax.scatter(x=xs, y=[x[0] for x in what])
        ax.get_yaxis().set_visible(False)
        ax.set_xscale("symlog")
        ax.set_xlim(DebugStats.make_range(xs))
        for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() +
                        ax.get_yticklabels()):
            item.set_fontsize(8)
        return fig

    def show(self):
        if not self.logger.is_currently_logging():
            return

        if len(self.acts) != 0:
            self.logger.log_mpl_figure("forward stats", self._show(self.acts))
        if len(self.bw_acts) != 0:
            self.logger.log_mpl_figure(
                "backward stats", self._show(list(reversed(self.bw_acts)))
            )

    def save(self, name, mod, mod_in, mod_out, bw=False):
        if bw:
            if isinstance(mod_out, tuple):
                mod_out = mod_out[0]

            m = mod_out.mean().detach().cpu().item()
            v = mod_out.var().detach().cpu().item()
            self.bw_acts.append([name, m, v])
        else:
            m = mod_out.mean().detach().cpu().item()
            v = mod_out.var().detach().cpu().item()
            self.acts.append([name, m, v])

    conv = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    convt = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
    convs = conv + convt
    bn = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    ln = (nn.LayerNorm,)
    inn = (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    norms = bn + inn + ln
    layer_dbg = (nn.Linear,) + convs + norms

    @staticmethod
    def attach(model, logger):
        dbg = DebugStats(logger)
        for nm, m in model.named_modules():
            if isinstance(m, DebugStats.layer_dbg):
                m.register_forward_hook(lambda *x, nm=nm: dbg.save(nm, *x))
                m.register_backward_hook(lambda *x, nm=nm: dbg.save(nm, *x, bw=True))
        return dbg


def log(msg, extra_msg='', other_keys=tuple(), level=logging.INFO, once=False):
    global log_msg_cache
    if log_msg_cache is None:
        log_msg_cache = set()

    msg_cache_key = (msg, *other_keys)
    if once and msg_cache_key in log_msg_cache:
        return

    logging.log(level, msg + ' ' + extra_msg)

    if once:
        log_msg_cache.add(msg_cache_key)


def edit_distance(x, y):
    """Returns the edit distance between sequences x and y. We are using
    dynamic programming to compute the minimal number of operations needed
    to transform sequence x into y."""
    dp = np.zeros((len(x) + 1, len(y) + 1), dtype="int64")
    for i in range(len(x) + 1):
        dp[i][0] = i
    for i in range(len(y) + 1):
        dp[0][i] = i
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # insertion
                dp[i][j - 1] + 1,  # deletion
                dp[i - 1][j - 1] + (0 if x[i - 1] == y[j - 1] else 1),
            )
    return dp[-1][-1]


def edit_distance_with_stats(x, y):
    dp = np.zeros((len(x) + 1, len(y) + 1), dtype="int64")
    op = np.zeros((len(x) + 1, len(y) + 1), dtype="int64")
    for i in range(len(x) + 1):
        dp[i][0] = i
        op[i][0] = 0
    for i in range(len(y) + 1):
        dp[0][i] = i
        op[0][i] = 1
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            operations = (
                dp[i - 1][j] + 1,  # insertion
                dp[i][j - 1] + 1,  # deletion
                dp[i - 1][j - 1] + (0 if x[i - 1] == y[j - 1] else 1),
            )
            choosen_op = np.argmin(operations)
            op[i][j] = choosen_op
            dp[i][j] = operations[choosen_op]
    i = len(x)
    j = len(y)
    operations = [0, 0, 0]
    while i >= 0 and j >= 0:
        print(i, j)
        old_op = op[i][j]
        ni = i if old_op == 1 else i - 1
        nj = j if old_op == 0 else j - 1
        if dp[i][j] > dp[ni][nj]:
            operations[old_op] += 1
        i = ni
        j = nj
        print(i, j)
    return (
        dp[-1][-1],
        {"ins": operations[0], "del": operations[1], "sub": operations[2]},
    )


"""
  NOTE: the example decoded sequence for 'tee' should include e.g.
  ' 0 0 0 e 0 e 0 0' i.e. removing repetitions
  does not remove 'ee' in decoded sequence.
"""


def remove_reps_blanks(preds):
    """remove duplicates, then blanks"""
    ret = []
    prev = -1
    for pred in preds:
        if pred == prev:
            continue
        prev = pred
        if prev != 0:
            ret.append(prev)
    return ret


def greedy_ctc_decode(log_probs, log_prob_lens, return_raw=False):
    # log_probs: (t x bs x nc)
    preds = log_probs.argmax(-1).to("cpu").int().numpy()
    bsz = log_prob_lens.size(0)
    decodes = [remove_reps_blanks(preds[: log_prob_lens[i], i]) for i in range(bsz)]
    if return_raw:
        decodes_raw = [preds[: log_prob_lens[i], i] for i in range(bsz)]
        return decodes, decodes_raw
    else:
        return decodes


def tensorList2list(h_):
    result = []
    for it in h_:
        result.append(it.item())
    return result


def error_rate(hyps, targets):
    verbose = 0

    assert len(hyps) == len(targets)
    tot_edits = 0.0
    tot_len = 0.0
    idx = 0
    for h, t in zip(hyps, targets):
        distance = editdistance.distance(np.array(h), np.array(t))

        if verbose > 0:
            # If necessary, get 'alphabet' as argument after which you can compare strings.
            # CHECK: Make sure no blanks/ class #0 in here
            print("error_rate() [" + str(idx) + "] hyps:    " + str(tensorList2list(h)))
            print("error_rate() [" + str(idx) + "] targets: " + str(tensorList2list(t)))
            print("error_rate() [" + str(idx) + "] distance: " + str(distance))

        tot_edits += distance
        tot_len += len(t)
        idx += 1
    # end for

    # Compute character error rate (CER) == label error rate (LER)
    cer = (tot_edits * 100.0) / tot_len

    return cer


def get_class(str_or_class, default_mod=None):
    if isinstance(str_or_class, six.string_types):
        parts = str_or_class.split(".")
        mod_name = ".".join(parts[:-1])
        class_name = parts[-1]
        if mod_name:
            mod = importlib.import_module(mod_name)
        elif default_mod is not None:
            mod = importlib.import_module(default_mod)
        else:
            raise ValueError("Specify a module for %s" % (str_or_class,))
        return getattr(mod, class_name)
    else:
        return str_or_class


def construct_from_kwargs(
    object_or_kwargs, default_mod=None, additional_parameters=None
):
    if not isinstance(object_or_kwargs, dict):
        assert not additional_parameters
        return object_or_kwargs
    object_kwargs = dict(object_or_kwargs)
    class_name = object_kwargs.pop("class_name")
    klass = get_class(class_name, default_mod)
    if additional_parameters:
        object_kwargs.update(additional_parameters)
    obj = klass(**object_kwargs)
    return obj


def uniq(inlist):
    """
    Behaves like UNIX uniq command - removes repeating items.
    Returns list of (start, end) pairs such that list[start:end] has only
    one distinct element
    """
    if inlist == []:
        return []

    outl = []
    current_start = 0
    current_element = inlist[0]
    for i, elem in enumerate(inlist[1:], start=1):
        if current_element != elem:
            outl.append((current_start, i))
            current_start = i
            current_element = elem
    outl.append((current_start, i + 1))
    return outl


def ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def conv_weights_xavier_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') == -1 and classname.find('Linear') == -1:
        return

    if hasattr(m, 'weight') and m.weight.requires_grad:
        try:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


def get_mask1d(lengths, mask_length=None, batch_first=True):
    """Get mask that is 1 for sequences shorter than lengths and 0 otherwise.

    The mask is on the device of lengths.
    """
    if mask_length is None:
        mask_length = lengths.max()
    lengths = lengths.long()
    arange = torch.arange(mask_length, device=lengths.device)
    if batch_first:
        mask = arange < lengths[:, None]
    else:
        mask = arange[:, None] < lengths
    return mask.float()


def get_mask2d(lengths, shape_as, mask_length=None, batch_first=True):
    m = get_mask1d(lengths, mask_length, batch_first)
    m = m.view(m.shape[0], 1, 1, m.shape[1])
    return m.expand_as(shape_as)


def get_mlp(num_inputs, hidden_dims, activation):
    net = torch.nn.Sequential()
    activation = getattr(torch.nn, activation)
    for i, dim in enumerate(hidden_dims):
        net.add_module(f"fc{i}", torch.nn.Linear(num_inputs, dim))
        net.add_module(f"act{i}", activation())
        num_inputs = dim
    return net, num_inputs


def extract_modify_dict(modify_config):
    if modify_config is None:
        return {}

    # Flatten
    modify_config = functools.reduce(lambda x,y: x+y, modify_config)
    even_list, odd_list = modify_config[::2], modify_config[1::2]

    if any(v.lower() == 'none' for v in odd_list):
        raise ValueError('Specify None values as "null"')

    if(len(even_list) != len(odd_list)):
        raise Exception(
            "Modify config list should have even number of elements")

    return dict(zip(even_list, odd_list))


# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def _atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [_atoi(c) for c in re.split(r"(\d+)", text)]


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def gather_nd_pt(table, k):
    # equivalent to tf.gather_nd
    return table.permute(2, 0, 1)[
        list((torch.arange(table.permute(2, 0, 1).size(0)), *k.chunk(2, 1)))
    ]


def compute_nmi(cluster_assignments, class_assignments):
    """Computes the Normalized Mutual Information between cluster and class assignments.
    Compare to https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

    Args:
        cluster_assignments (list): List of cluster assignments for every point.
        class_assignments (list): List of class assignments for every point.
    Returns:
        float: The NMI value.
    """
    assert len(cluster_assignments) == len(
        class_assignments
    ), "The inputs have to be of the same length."

    clusters = np.unique(cluster_assignments)
    classes = np.unique(class_assignments)

    num_samples = len(cluster_assignments)
    num_clusters = len(clusters)
    num_classes = len(classes)

    assert num_classes > 1, "There should be more than one class."

    cluster_class_counts = {
        cluster_: {class_: 0 for class_ in classes} for cluster_ in clusters
    }

    for cluster_, class_ in zip(cluster_assignments, class_assignments):
        cluster_class_counts[cluster_][class_] += 1

    cluster_sizes = {
        cluster_: sum(list(class_dict.values()))
        for cluster_, class_dict in cluster_class_counts.items()
    }
    class_sizes = {
        class_: sum([cluster_class_counts[clus][class_] for clus in clusters])
        for class_ in classes
    }

    I_cluster_class = H_cluster = H_class = 0

    for cluster_ in clusters:
        for class_ in classes:
            if cluster_class_counts[cluster_][class_] == 0:
                pass
            else:
                I_cluster_class += (
                    cluster_class_counts[cluster_][class_] / num_samples
                ) * (
                    np.log(
                        (cluster_class_counts[cluster_][class_] * num_samples)
                        / (cluster_sizes[cluster_] * class_sizes[class_])
                    )
                )

    for cluster_ in clusters:
        H_cluster -= (cluster_sizes[cluster_] / num_samples) * np.log(
            cluster_sizes[cluster_] / num_samples
        )

    for class_ in classes:
        H_class -= (class_sizes[class_] / num_samples) * np.log(
            class_sizes[class_] / num_samples
        )

    NMI = (2 * I_cluster_class) / (H_cluster + H_class)

    return NMI


def compute_purity(cluster_assignments, class_assignments):
    """Computes the purity between cluster and class assignments.
    Compare to https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

    Args:
        cluster_assignments (list): List of cluster assignments for every point.
        class_assignments (list): List of class assignments for every point.
    Returns:
        float: The purity value.
    """
    assert len(cluster_assignments) == len(class_assignments)

    num_samples = len(cluster_assignments)
    num_clusters = len(np.unique(cluster_assignments))
    num_classes = len(np.unique(class_assignments))

    cluster_class_counts = {
        cluster_: {class_: 0 for class_ in np.unique(class_assignments)}
        for cluster_ in np.unique(cluster_assignments)
    }

    for cluster_, class_ in zip(cluster_assignments, class_assignments):
        cluster_class_counts[cluster_][class_] += 1

    total_intersection = sum(
        [
            max(list(class_dict.values()))
            for cluster_, class_dict in cluster_class_counts.items()
        ]
    )

    purity = total_intersection / num_samples

    return purity


def ptvsd(host="0.0.0.0", port=5678):
    import ptvsd

    ptvsd.enable_attach(address=(host, port), redirect_output=True)
    print(f"PTVSD waiting for attach at {host}:{port}")
    ptvsd.wait_for_attach()
    breakpoint()


def sample_truncated_normal(starts, ends, stddev):
    dist = torch.distributions.Normal(
        torch.tensor([0.0]), torch.tensor([math.sqrt(stddev)])
    )
    batch_size = len(starts)
    sample = dist.sample((batch_size,)).squeeze()
    middle = starts.float() + (ends - starts).float() / 2.0
    untruncated_sampled_positions = (torch.round(middle + sample)).long()
    return torch.min(torch.max(starts, untruncated_sampled_positions), ends)


def reverse_sequences(mini_batch, seq_lengths):
    reversed_mini_batch = torch.zeros_like(mini_batch)
    for b in range(mini_batch.size(0)):
        T = seq_lengths[b]
        time_slice = torch.arange(T - 1, -1, -1, device=mini_batch.device)
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
    return reversed_mini_batch


def pad_and_reverse(rnn_output, seq_lengths):
    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    reversed_output = reverse_sequences(rnn_output, seq_lengths)
    return reversed_output


def get_mini_batch_mask(mini_batch, seq_lengths):
    mask = torch.zeros(mini_batch.shape[0:2])
    for b in range(mini_batch.shape[0]):
        mask[b, 0 : seq_lengths[b]] = torch.ones(seq_lengths[b])
    return mask


def calc_au(means, VAR_THRESHOLD=1e-2):
    # get number of active units
    z_means = torch.cat(means, dim=0)
    var_z = torch.std(z_means, dim=0).pow(2)
    active_units = torch.arange(0, z_means.size(1))[var_z > VAR_THRESHOLD].long()
    n_active_z = len(active_units)
    return n_active_z, active_units


def rleEncode(x):
    """Run length encoding of a 1D torch tensor.
    Input is like [ 0 0 0 0 0 1 1 3 3 3 3]
    and output a 2d tensors of pairs (class,length)  like [ [ 0,5] [1,2] [3,4] ]
    # TEST as : print ( distsup.utils.rleEncode( torch.tensor([ 0, 0, 0, 0, 0, 1, 1, 3, 3, 3, 3]) ))
    """
    assert len(x.shape) == 1

    where = np.flatnonzero
    n = len(x)

    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    result = torch.IntTensor(
        [[starts[i], starts[i] + lengths[i] - 1] for i in range(len(starts))]
    )

    return result, values

def _make_liftering(N, Q):
    return 1 + 0.5*Q*np.sin(np.pi*np.arange(N)/Q).astype(np.float32)


def _make_dct(input_dim, output_dim, inv, normalize):
    from scipy.fftpack import dct, idct
    if normalize:
        norm = 'ortho'
    else:
        norm = None
    if inv:
        C = idct(np.eye(input_dim), type=2, norm=norm, overwrite_x=True)
    else:
        C = dct(np.eye(input_dim), type=2, norm=norm, overwrite_x=True)
    return C[:,:output_dim].astype(np.float32)

class FBANK2MFCC(nn.Module):
    def __init__(self, input_dim, keep_energy=True):
        super(FBANK2MFCC, self).__init__()
        input_dim = input_dim - 1
        self.lift = nn.Parameter(torch.from_numpy(_make_liftering(input_dim, input_dim-1)), requires_grad=False)
        self.dct  = nn.Parameter(torch.from_numpy(_make_dct(input_dim, input_dim, inv=False, normalize=True)), requires_grad=False)
        self.keep_energy = keep_energy

    def forward(self, x):
        x = x.permute(0, 1, 3, 2)
        x[:, :, :, 1:] = nn.functional.linear(x[:, :, :, 1:], self.dct) * self.lift
        if not self.keep_energy:
            x = x[:, :, :, 1:]
        return x[:, :, :1, :].permute(0, 1, 3, 2)


# https://github.com/allenai/allennlp/blob/30c4271f7f04babb1cb546ab017a104bda011e7c/allennlp/nn/util.py#L376
def masked_flip(padded_sequence, sequence_lengths):
    """
        Flips a padded tensor along the time dimension without affecting masked entries.
        Parameters
        ----------
        padded_sequence : ``torch.Tensor``
            The tensor to flip along the time dimension.
            Assumed to be of dimensions (batch size, num timesteps, ...)
        sequence_lengths : ``torch.Tensor``
            A list containing the lengths of each unpadded sequence in the batch.
        Returns
        -------
        A ``torch.Tensor`` of the same shape as padded_sequence.
        """
    assert padded_sequence.size(0) == len(sequence_lengths), \
        f'sequence_lengths length ${len(sequence_lengths)} does not match batch size ${padded_sequence.size(0)}'
    num_timesteps = padded_sequence.size(1)
    flipped_padded_sequence = torch.flip(padded_sequence, [1])
    sequences = [flipped_padded_sequence[i, num_timesteps - length:] for i, length in enumerate(sequence_lengths)]
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)


def corrupt(x, p):
    '''Zeroes values with probability p'''
    ber = torch.distributions.bernoulli.Bernoulli(
        torch.tensor([1.0 - p], device=x.device))
    mask = safe_squeeze(ber.sample(sample_shape=x.size()), -1)
    # mask = torch.empty_like(x).uniform_() > self.gap_corruption
    return x * mask

