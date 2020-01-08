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

import numpy as np

import openfst_python as fst

import torch

from distsup.utils import safe_squeeze

#
# Functions to build FST graphs
#


def build_chain_fst(labels, arc_type='log', vocab=None):
    """
    Build an acceptor for string given by elements of labels.
    Args:
        labels - a sequence of labels in the range 1..S
        arc_type - fst arc type (standard or log)
    Returns:
        FST consuming symbols in the range 1..S.
    Notes:
        Elements of labels are assumed to be greater than zero
        (which maps to blank)!
    """
    C = fst.Fst(arc_type=arc_type)
    weight_one = fst.Weight.One(C.weight_type())
    s = C.add_state()
    C.set_start(s)
    for l in labels:
        s_next = C.add_state()
        C.add_arc(s, fst.Arc(l, l, weight_one, s_next))
        s = s_next
    C.set_final(s)
    C.arcsort('ilabel')
    return C


def fst_to_matrices(g, out_edges=True, nc_weight=None, device=None):
    """
    Encode FST transitions as adjacency lists, prepared as padded matrices.
    Args:
        g - an FST. It shall not have epsilons on input side,
            the outputs are ignored. The ilabels are treated as
            (1 + input symbol)
        out_edges - if True, for each node enumerate outgoing arcs,
                    if False, enumerate incoming arcs
        nc_weight - weight ot assign to masked edges (no connections).
            Should be about -1e20 for single floating precision
        device - device on which to place the tensors
    Returns
        Let K be the maximum out-degree (in-degree when not out_edges), and
        N number of nodes in the FST. 4 tensors are returned:
        states_mat: N x K matrix of successor (predecessor) state id
        ilabels_mat: N x K, labels consumed on the edges
        weights_mat: N x K, log-prob of the edge. Equals to nc_weight if
            the edge is introduced for padding
        terminal_mat: N x 1, log-probs of terminal noeds
            (and neg_inf if node is not terminal)
    """
    if g.start() != 0:
        raise ValueError("FST starting state is not 0, but %d" %
                         (g.start(),))

    is_det = (g.properties(fst.I_DETERMINISTIC, fst.I_DETERMINISTIC) > 0)
    if not is_det:
        raise ValueError("FST is not deterministic")

    if nc_weight is None:
        nc_weight = -fst.Weight.Zero(g.weight_type())
    nc_weight = float(nc_weight)

    edges = [[] for _ in g.states()]
    n = g.num_states()
    terminal_mat = torch.full((n, 1), nc_weight, dtype=torch.float32)
    for prevstate in g.states():
        assert prevstate < n
        term_weight = -float(g.final(prevstate))
        if np.isfinite(term_weight):
            terminal_mat[prevstate, 0] = term_weight
        else:
            terminal_mat[prevstate, 0] = nc_weight

        for a in g.arcs(prevstate):
            ilabel = a.ilabel - 1
            weight = -float(a.weight)
            if ilabel < 0:
                raise ValueError(
                    "FST has eps-transitions (state=%d)" % (prevstate,))

            if out_edges:
                edges[prevstate].append((a.nextstate, ilabel, weight))
            else:
                edges[a.nextstate].append((prevstate, ilabel, weight))
    k = max(len(e) for e in edges)
    states_mat = torch.full((n, k), 0, dtype=torch.int64)
    ilabels_mat = torch.full((n, k), 0, dtype=torch.int64)
    weights_mat = torch.full((n, k), nc_weight, dtype=torch.float32)
    for s1, arcs in enumerate(edges):
        for i, (s2, ilabel, weight) in enumerate(sorted(arcs)):
            states_mat[s1, i] = s2
            ilabels_mat[s1, i] = ilabel
            weights_mat[s1, i] = weight
    if device is not None:
        states_mat = states_mat.to(device)
        ilabels_mat = ilabels_mat.to(device)
        weights_mat = weights_mat.to(device)
        terminal_mat = terminal_mat.to(device)
    return states_mat, ilabels_mat, weights_mat, terminal_mat


#
# Viterbi and forward-backward
#


def path_reduction(
        log_probs, act_lens, graph_matrices, red_kind='logsumexp',
        neg_inf=-1e20):
    """
    Compute a sum of all paths through a graph.
    Args:
        log_probs: bs x T x 1 x NUM_SYMBOLS tensor of log_probs of emitting symbols
        act_lens: bs tensor of lengths of utternaces
        red_kind: logsumexp / viterbi - chooses between aggregating al paths by
            summing their probabilities (logsumexp of logprobs), or
            by taking the maximally probable one. Also encoded which reduction
            engige ot use:
                logsumexp_fwb forces a forward-backward algo, while
                logsumexp_autodiff uses backward pass using autodiff.
        graphs_matrices: a tuple of four matrices of shape bs x N [x K]
            that encode the transitions and weights in the graph
        neg_inf: what value to use for improbable events (-1e10 or -1e20 are OK)
    Returns:
        tensor of shape bs: a sum of weigths on the maximally probable path
        or on all paths
    """
    if (red_kind == 'logsumexp_fwb' or
            (red_kind == 'logsumexp' and len(graph_matrices) == 8)):
        return path_logsumexp(log_probs, act_lens, graph_matrices, -1e20)

    log_probs = safe_squeeze(log_probs, 2).transpose(0, 1).contiguous()
    _, bs, _ = log_probs.size()
    assert graph_matrices[0].size(0) in [1, bs]
    assert all(sm.size(0) == graph_matrices[0].size(0)
               for sm in graph_matrices)
    # This can happen if we get the matrices for full forward-backward
    # and here we only need the ones for worward
    if len(graph_matrices) == 8:
        graph_matrices = graph_matrices[:4]
    if graph_matrices[0].size(0) == 1:
        graph_matrices = [gm.expand(bs, -1, -1) for gm in graph_matrices]
    states_mat, ilabels_mat, weights_mat, terminal_mat = graph_matrices

    _, n, k = states_mat.size()

    if red_kind in ['logsumexp', 'logsumexp_autodiff']:
        # reduction = torch.logsumexp
        reduction = torch.logsumexp
    else:
        assert red_kind in ['viterbi', 'viterbi_autodiff']

        def reduction(t, dim):
            return torch.max(t, dim)[0]

    # a helper to select the next indices for a transition
    def get_idx(m, i):
        _bs = m.size(0)
        return torch.gather(m, 1, i.view(_bs, n * k)).view((_bs, n, k))

    lalpha = torch.full((bs, n), neg_inf, device=log_probs.device)
    lalpha[:, 0] = 0

    # The utterances are sorted according to length descending.
    # Rather than masking, stop updates to alphas when an utterance ends.
    assert act_lens.tolist() == sorted(act_lens, reverse=True)
    last_iter_end = 0
    for bitem in range(bs, 0, -1):
        iter_end = act_lens[bitem - 1]
        for t in range(last_iter_end, iter_end):
            # print(torch.softmax(lalpha[0], -1))
            token_probs = (
                get_idx(lalpha[:bitem], states_mat[:bitem]) +
                weights_mat[:bitem] +
                get_idx(log_probs[t, :bitem], ilabels_mat[:bitem]))
            la = reduction(token_probs, dim=-1)
            lalpha = lalpha.clone()
            lalpha[:bitem] = la
        last_iter_end = iter_end

    path_sum = reduction(lalpha + terminal_mat.squeeze(2), dim=-1)
    return path_sum


class PathLogSumExp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, log_probs, act_lens, graph_matrices, neg_inf=-np.inf):
        logsumexp = torch.logsumexp
        log_probs = log_probs.detach()
        log_probs = safe_squeeze(log_probs, 2).transpose(0, 1).contiguous()
        T, bs, _ = log_probs.size()
        assert graph_matrices[0].size(0) in [1, bs]
        assert all(sm.size(0) == graph_matrices[0].size(0)
                   for sm in graph_matrices)
        if graph_matrices[0].size(0) == 1:
            graph_matrices = [gm.expand(bs, -1, -1) for gm in graph_matrices]
        (states_mat, ilabels_mat, weights_mat, terminal_mat,
         states_mat_out, ilabels_mat_out, weights_mat_out, _
         ) = graph_matrices

        terminal_mat = terminal_mat.squeeze(-1)

        _, n, _ = states_mat.size()

        # a helper to select the next indices for a transition
        def get_idx(m, i):
            _bs = m.size(0)
            return torch.gather(m, 1, i.view(_bs, -1)).view(i.size())

        lalpha = torch.full((bs, n), neg_inf, device=log_probs.device)
        lalpha[:, 0] = 0
        lalpha0 = lalpha.clone()

        lalphas = torch.full((T, bs, n), neg_inf, device=log_probs.device)

        # The utterances are sorted according to length descending.
        # Rather than masking, stop updates to alphas when an utterance ends.
        assert act_lens.tolist() == sorted(act_lens, reverse=True)
        last_iter_end = 0
        for bitem in range(bs, 0, -1):
            iter_end = act_lens[bitem - 1]
            for t in range(last_iter_end, iter_end):
                lalphas[t] = lalpha
                token_probs = weights_mat[:bitem].clone()
                token_probs += get_idx(lalpha[:bitem], states_mat[:bitem])
                token_probs += get_idx(log_probs[t, :bitem],
                                       ilabels_mat[:bitem])
                logsumexp(token_probs, dim=-1, out=lalpha[:bitem])
            last_iter_end = iter_end

        log_cost = logsumexp(lalpha + terminal_mat, dim=-1)

        lbeta = terminal_mat.clone()
        logprobs_grad = torch.zeros_like(log_probs)

        last_iter_end = T
        for bitem in range(1, bs + 1):
            if bitem < bs:
                iter_end = act_lens[bitem]
            else:
                iter_end = 0
            for t in range(last_iter_end - 1, iter_end - 1, -1):
                token_probs = weights_mat_out[:bitem].clone()
                token_probs += get_idx(lbeta[:bitem], states_mat_out[:bitem])
                token_probs += get_idx(log_probs[t, :bitem],
                                       ilabels_mat_out[:bitem])
                logsumexp(token_probs, dim=-1, out=lbeta[:bitem])

                token_probs += (lalphas[t, :bitem] -
                                log_cost[:bitem].unsqueeze(-1)
                                ).unsqueeze(-1)
                token_probs.exp_()

                logprobs_grad[t, :bitem].scatter_add_(
                    1, ilabels_mat_out[:bitem].view(bitem, -1),
                    token_probs.view(bitem, -1))
            last_iter_end = iter_end

        ctx.grads = logprobs_grad.transpose(0, 1).unsqueeze(2)

        # approximate the numerical error
        log_cost0 = logsumexp(lalpha0 + lbeta, dim=1)
        if torch.abs(log_cost - log_cost0).max().item() > 1e-3:
            print('forward_backward num error: fwd losses %s bwd losses %s' %
                  (log_cost, log_cost0))
        return log_cost

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output[:, None, None, None] * ctx.grads,
                None, None, None, None, None, None)


path_logsumexp = PathLogSumExp.apply


def batch_training_graph_matrices(matrices,
                                  nc_weight=-1e20, device='cpu'):
    """
    Combine training matrices for a batch of labels.
    Args:
        matrices: list of tuples of matrices for FSTs
        nc_weight - forarded to fst_to_matrices
        device: pytorch device
    Returns:
        The matrices (see fst_to_matrices) for training FSTs (CTC graphs
        that accept only the trainig utterance), concatenated to a large
        padded matrix of size B x N x K, where B is the batch size,
        N maximum number of states, K maximum degree.
    """
    bs = len(matrices)
    max_n = max([m[0].size(0) for m in matrices])
    max_ks = [max([m[i].size(1) for m in matrices])
              for i in range(len(matrices[0]))]
    batched_matrices = []
    for i, m in enumerate(matrices[0]):
        batched_matrices.append(torch.full(
            (bs, max_n, max_ks[i]),
            0 if m.dtype == torch.int64 else nc_weight,
            dtype=m.dtype, device=device
            ))

    for b, ms in enumerate(matrices):
        for i, m in enumerate(ms):
            batched_matrices[i][b, :m.size(0), :m.size(1)] = m
    return batched_matrices
