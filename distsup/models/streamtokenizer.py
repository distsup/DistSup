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

import copy
import logging
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from distsup.logger import default_tensor_logger
from distsup.models import probednet
from distsup import scoring, utils
import os

logger = default_tensor_logger.DefaultTensorLogger()


class StreamTokenizerNet(probednet.ProbedNet):
    """
    A model that takes a stream of features (maybe real valued) and
    produces a sequence of discrete tokens time-aligned with the stream.
    Note that the rate may be different but remains a constant.

    This model performs several evaluations:
     - perplexity (usage of the tokens)
     - adjusted rand index (if alignment groundtruth is available in the batch)
     - greedy mapping
     - probes

     In order to use probes you must add the `probes` element in the model.
     It must be a dictionary as follows:

     ```
       probes:
         vq_pred:
           layer: bottleneck
           target: alignment
           predictor:
             class_name: distsup.modules.predictors.FramewisePredictor
             aggreg: 3
           bp_to_main: False
     ```
    """
    pad_symbol = 0

    def __init__(self, **kwargs):
        super(StreamTokenizerNet, self).__init__(**kwargs)

    @staticmethod
    def plot_input_and_alignments(features, reconstruction=None,
                                  alignment_es=None, alignment_gt=None,
                                  mapping=None,
                                  imshow_kwargs=None, log_suffix=''):
        imshow_kwargs = imshow_kwargs if imshow_kwargs is not None else {}
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            'font.size': 3,
            'xtick.labelsize': 'x-small',
            'ytick.labelsize': 'x-small',
        })

        features = features.detach()[0, :, :, 0].permute(1, 0)
        feats = features.cpu().numpy()

        n_subplots = 2 if reconstruction is not None else 1

        f, axs = plt.subplots(n_subplots, sharex='all', squeeze=False, dpi=300,
                              figsize=(6.4, 1.2))

        ax = axs[0][0]
        # ax = f.add_subplot(n_subplots, 1, 1)
        ax.imshow(feats, **imshow_kwargs)
        ax.set_title('input features')

        def _plot_alignment(feat, alignment, axis,
                            text_y=-0.01,
                            axvline_kwargs=None):
            axvline_kwargs = axvline_kwargs if axvline_kwargs is not None else {}
            trans = axis.get_xaxis_transform()
            scale = int(np.round(feat.size(1) / alignment.size(1)))
            i = 0
            indices = alignment.cpu().long()
            while i < indices.size(1):
                j = i
                while j + 1 < indices.size(1) and indices[0, j + 1] == indices[0, i]:
                    j += 1
                tok_id = indices[0, j].cpu().item()
                axis.text(j * scale / 2. + i * scale / 2. + scale / 2., text_y,
                          str(tok_id), rotation=45, transform=trans,
                          horizontalalignment='center', verticalalignment='baseline',
                          color='red')
                axis.axvline(j * scale + scale, **axvline_kwargs)
                i = j + 1

        if alignment_es is not None:
            _plot_alignment(features, alignment_es, ax,
                            text_y=-0.01,
                            axvline_kwargs=dict(linewidth=1.0,
                                                linestyle='-',
                                                ymin=0,
                                                ymax=0.5))

        if alignment_gt is not None:
            _plot_alignment(features, alignment_gt, ax,
                            text_y=1.01,
                            axvline_kwargs=dict(linewidth=1.0,
                                                linestyle='-',
                                                ymin=0.5,
                                                ymax=1.0))

        if (mapping is not None) and (alignment_es is not None) and (alignment_gt is not None):
            # TODO: show the correctly and incorrectly mapped sequences
            reduction = int(np.round(features.size(1) / alignment_es.size(1)))
            alignment_es = torch.repeat_interleave(alignment_es.detach().cpu(), reduction, dim=1)

            alignment_es_mapped = alignment_es[0].numpy().copy()
            alignment_gt = alignment_gt[0].detach().cpu().numpy()

            alignment_es_mapped = alignment_es_mapped[:, 0, 0]

            for src, tgt in mapping.items():
                alignment_es_mapped[alignment_es_mapped == src] = tgt

            min_len = min(alignment_gt.shape[0], alignment_es_mapped.shape[0])
            alignment_es_mapped = alignment_es_mapped[:min_len]
            alignment_gt = alignment_gt[:min_len]

            correct = (alignment_es_mapped == alignment_gt)

            starts_ends, values = utils.rleEncode(correct)
            starts_ends = starts_ends.numpy()

            starts_ends = starts_ends[values]
            xs = np.c_[starts_ends[:, 0], (starts_ends[:, 1] + 1) - starts_ends[:, 0]]
            ys = list(sorted(list(ax.get_ylim())))

            ax.broken_barh(xs, ys,
                           facecolors='tab:green',
                           alpha=0.2)

        if reconstruction is not None:
            reconstruction = reconstruction[0, :, :, 0].permute(1, 0)
            # ax_recons = f.add_subplot(n_subplots, 1, 2, sharex=ax)
            ax_recons = axs[1][0]
            ax_recons.imshow(reconstruction.cpu().numpy(), **imshow_kwargs)

        f.set_tight_layout(True)
        logger.log_mpl_figure(f'segmentation_{log_suffix}', f)
        plt.close(f)

    def align_tokens_to_features(self, batch, tokens):
        """
        This method should produce a version of the tokens so that each
        feature timestep has a corresponding token.
        This depends on the encoder (e.g. padding, downsampling, etc.)
        This method will be used by the evaluation and visualisation.

        The return value must be a tensor of shape (B, W)
        """
        raise NotImplementedError

    def minibatch_loss_and_tokens(self, batch):
        """
        This method performs the forward pass and returns the loss, the
        stats dictionary and the sequence of tokens.

        The return value must be a tuple: (loss, details, tokens)
        Where tokens can be None if the model does not output any.
        """
        raise NotImplementedError

    def minibatch_loss(self, batch):
        loss, stats, tokens = self.minibatch_loss_and_tokens(batch)
        detached_loss, backprop_loss, probes_details = self.probes_loss(batch)
        stats.update(probes_details)
        stats['probes_detached_loss'] = detached_loss
        stats['probes_backprop_loss'] = backprop_loss
        stats['probes_loss'] = detached_loss + backprop_loss

        if tokens is not None:
            self.plot_input_and_alignments(batch['features'],
                                           alignment_es=tokens,
                                           alignment_gt=batch.get('alignment', None),
                                           mapping=None,
                                           imshow_kwargs=dict(cmap='Greys'))

        return loss + detached_loss + backprop_loss, stats

    @staticmethod
    def _clustering_metrics(ali_gt, ali_es, prefix=''):
        import sklearn.metrics
        return {
            f'{prefix}adjusted_mutual_info':
                sklearn.metrics.adjusted_mutual_info_score(ali_gt,
                                                           ali_es,
                                                           average_method='arithmetic'),
            f'{prefix}normalized_mutual_info':
                sklearn.metrics.normalized_mutual_info_score(ali_gt,
                                                             ali_es,
                                                             average_method='arithmetic'),
            f'{prefix}adjusted_rand_score':
                sklearn.metrics.adjusted_rand_score(ali_gt,
                                                    ali_es)
        }

    @staticmethod
    def _mapping_metrics(ali_gt, ali_es, prefix=''):
        # Many-to-one mapping from estimated to groundtruth
        es_uniq = np.unique(ali_es)

        mapping = {}
        alis_es_mapped = np.empty_like(ali_es)
        for es_sym in es_uniq:
            gt_syms, gt_counts = np.unique(ali_gt[ali_es == es_sym], return_counts=True)
            mapped_sym = gt_syms[np.argmax(gt_counts)]
            mapping[es_sym] = mapped_sym
            alis_es_mapped[ali_es == es_sym] = mapped_sym

        # one-to-one mapping accuracy with `linear_sum_assignment`
        mapping_size = np.concatenate(
            [np.unique(ali_gt), es_uniq]
        ).max() + 1
        mappings_cost = np.zeros((mapping_size, mapping_size))
        one_to_one_mapped = np.empty_like(ali_es)
        for i in range(mapping_size):
            gt_syms, gt_counts = np.unique(
                ali_gt[ali_es == i],
                return_counts=True
            )
            mappings_cost[i][gt_syms] = -gt_counts

        map_row, map_col = linear_sum_assignment(mappings_cost)
        for es_sym, mapped_sym in zip(map_row, map_col):
            one_to_one_mapped[ali_es == es_sym] = mapped_sym

        return (
            {
                f'many_es_to_one_gt_accuracy/{prefix}': (
                    (alis_es_mapped == ali_gt).mean()
                ),
                f'one_es_to_one_gt_accuracy/{prefix}': (
                    (one_to_one_mapped == ali_gt).mean()
                ),
            },
            mapping
        )

    @staticmethod
    def _perplexity_metrics(ali_es, prefix=''):
        # Compute the perplexity as the
        sym, counts = np.unique(ali_es, return_counts=True)

        p = counts / counts.sum()

        perplexity = 2**(-np.sum(p * np.log2(p)))

        return {f'{prefix}perplexity': perplexity,
                f'{prefix}used_tokens': len(sym)}

    @staticmethod
    def _unpad_and_concat(alis, alis_len):
        alis_unp = []

        # remove the padding tokens both from groundtruth and estimated alignments
        for ali_batch, ali_len_batch in zip(alis, alis_len):
            for sample_ali, sample_ali_len in zip(ali_batch, ali_len_batch):
                if len(sample_ali) < sample_ali_len:
                    raise ValueError(f'Expected the length of the alignment ({len(sample_ali)}) to be at '
                                     f'least as long as the groundtruth length before padding ({sample_ali_len}). '
                                     f'Has the method StreamTokenizer.align_tokens_to_features() been '
                                     f'properly implemented?')
                alis_unp.append(sample_ali[:sample_ali_len].detach().cpu().numpy())

        alis_unp = np.concatenate(alis_unp).astype(np.int)

        return alis_unp

    def pre_evaluate(self, batches):
        # TODO: Try first training the probes on a dev set
        pass

    def evaluate(self, batches):
        tot_examples = 0.
        tot_loss = 0.
        tot_detached_probesloss = 0.
        tot_backprop_probesloss = 0.
        tot_errs = 0.

        alis_es = []
        alis_gt = []
        alis_lens = []
        total_stats = {}

        first_batch = None

        for batch in batches:
            if first_batch is None:
                first_batch = copy.deepcopy(batch)

            num_examples = batch['features'].shape[0]
            loss, stats, tokens = self.minibatch_loss_and_tokens(batch)

            # Run the probes
            detached_loss, backprop_loss, probes_details = self.probes_loss(batch)
            stats.update(probes_details)

            if tokens is not None:
                # Tokens should be in layout B x W x 1 x 1
                tokens = utils.safe_squeeze(tokens, dim=3)
                tokens = utils.safe_squeeze(tokens, dim=2)

                feat_len = batch['features_len']
                alis_lens.append(feat_len)

                # the tokens should match the rate of the alignment
                ali_es = self.align_tokens_to_features(batch, tokens)
                assert(ali_es.shape[0] == batch['features'].shape[0])
                assert(ali_es.shape[1] == batch['features'].shape[1])
                alis_es.append(ali_es[:, :])
                if 'alignment' in batch:
                    ali_gt = batch['alignment']
                    ali_len = batch['alignment_len']

                    assert((ali_len == feat_len).all())
                    alis_gt.append(ali_gt)

            tot_examples += num_examples
            tot_loss += loss * num_examples
            tot_errs += stats.get('err', np.nan) * num_examples

            tot_detached_probesloss += detached_loss * num_examples
            tot_backprop_probesloss += backprop_loss * num_examples
            for k, v in stats.items():
                if k == 'segmental_values':
                    if logger.is_currently_logging():
                        import matplotlib.pyplot as plt
                        f = plt.figure(dpi=300)
                        plt.plot(v.data.cpu().numpy(), 'r.-')
                        f.set_tight_layout(True)
                        logger.log_mpl_figure(f'segmentation_values', f)
                elif utils.is_scalar(v):
                    if k not in total_stats:
                        total_stats[k] = v * num_examples
                    else:
                        total_stats[k] += v * num_examples
        # loss is special, as we use it e.g. for learn rate control
        # add all signals that we train agains, but remove the passive ones
        all_scores = {'loss': (tot_loss + tot_backprop_probesloss) / tot_examples,
                      'probes_backprop_loss': tot_backprop_probesloss / tot_examples,
                      'probes_detached_loss': tot_detached_probesloss / tot_examples,
                      'err': tot_errs / tot_examples,
                      'probes_loss': (tot_detached_probesloss + tot_backprop_probesloss
                                      ) / tot_examples
                      }

        for k, v in total_stats.items():
            all_scores[k] = v / tot_examples

        if (len(alis_es) > 0) and (len(alis_gt) > 0):
            # If we have gathered any alignments
            f1_scores = dict(precision=[], recall=[], f1=[])
            for batch in zip(alis_gt, alis_es, alis_lens):
                batch = [t.detach().cpu().numpy() for t in batch]
                for k,v in scoring.compute_f1_scores(*batch, delta=1).items():
                    f1_scores[k].extend(v)
            for k in ('f1', 'precision', 'recall'):
                print(f"f1/{k}: {np.mean(f1_scores[k])}")
                logger.log_scalar(f'f1/{k}', np.mean(f1_scores[k]))

            alis_es = self._unpad_and_concat(alis_es, alis_lens)
            alis_gt = self._unpad_and_concat(alis_gt, alis_lens) if len(alis_gt) else None

            scores_to_compute = [('', lambda x: x)]
            if alis_gt is not None and self.pad_symbol is not None:
                not_pad = (alis_gt != self.pad_symbol)
                scores_to_compute.append(('nonpad_', lambda x: x[not_pad]))

            if alis_gt is not None and alis_es.min() < 0:
                not_pad2 = (alis_es != -1)
                scores_to_compute.append(('validtokens_', lambda x: x[not_pad2]))

            for prefix, ali_filter in scores_to_compute:
                es = ali_filter(alis_es)

                if alis_gt is not None:
                    gt = ali_filter(alis_gt)

                    mapping_scores, mapping = self._mapping_metrics(gt, es, prefix=prefix)
                    all_scores.update(mapping_scores)

                    # Run the segmentation plottin with mapping
                    if logger.is_currently_logging():
                        _, _, tokens = self.minibatch_loss_and_tokens(first_batch)
                        self.plot_input_and_alignments(first_batch['features'],
                                                       alignment_es=tokens,
                                                       alignment_gt=first_batch['alignment'],
                                                       mapping=mapping,
                                                       imshow_kwargs=dict(cmap='Greys'),
                                                       log_suffix=f'{prefix[:-1]}')

                    clustering_scores = self._clustering_metrics(gt, es, prefix=prefix)
                    all_scores.update(clustering_scores)

                perplexity_scores = self._perplexity_metrics(es, prefix=prefix)
                all_scores.update(perplexity_scores)

        return all_scores


def main():
    pass


if __name__ == '__main__':
    main()
