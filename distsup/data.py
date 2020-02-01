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


from collections import OrderedDict, namedtuple, Counter
import bisect
import logging
import mmap
import os
import os.path
import pickle
import random
import hashlib

import numpy as np

import torch.utils.data
import torch.nn.functional as F

import kaldi_io
import cdblib

from distsup import utils
from distsup.alphabet import Alphabet
from distsup.configuration import Globals

mod_logger = logging.getLogger(__name__)


class CDBDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        super(CDBDataset, self).__init__()
        self.root = root
        fp = open(root, 'rb')
        mem = mmap.mmap(fp.fileno(), os.path.getsize(fp.name), mmap.MAP_SHARED,
                mmap.PROT_READ)
        self.reader = cdblib.Reader(mem)
        self.len = self.reader.getint(b'len')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return pickle.loads(self.reader.get(bytes(idx)))



class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, real_class_name, **kwargs):
        super(CachedDataset, self).__init__()
        self._wrapped = utils.construct_from_kwargs(
            {"class_name": real_class_name}, additional_parameters=kwargs
        )
        self._cache = [None] * len(self._wrapped)

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, item):
        if self._cache[item] is None:
            self._cache[item] = self._wrapped[item]
        return self._cache[item]

    def __getattr__(self, attr):
        return getattr(self._wrapped, attr)


class SequentialFrameDataset(torch.utils.data.Dataset):
    """A dataset that converts a sequential dataset to a dataset with
       individual 'frames' and the corresponding targets.
       It should have fields alignment_rle and text.
    """

    def __init__(self, dataset, chunk_len, feature_field="features", every_n=1):
        self.dataset = utils.construct_from_kwargs(dataset)
        self.chunk_len = chunk_len
        self.feature_field = feature_field
        self.alphabet = self.dataset.alphabet
        self.every_n = every_n

        self.item_id_shift = []

        self.all_targets = []
        for i in range(len(self.dataset)):
            self.item_id_shift.append(len(self.all_targets))
            item = self.dataset[i]
            text = item["text"]
            alignment_rle = item["alignment_rle"]
            data = item[self.feature_field]

            for unit_id, (start, end) in zip(text, alignment_rle):
                for pos in range(int(start), int(end + 1)):
                    if (
                        pos - self.chunk_len // 2 >= 0
                        and pos - self.chunk_len // 2 + self.chunk_len - 1 < len(data)
                    ):
                        self.all_targets.append(unit_id)

        self.all_targets = torch.tensor(self.all_targets)

    def __len__(self):
        return len(self.all_targets) // self.every_n

    def __getitem__(self, idx):
        idx *= self.every_n
        item_id = bisect.bisect(self.item_id_shift, idx) - 1
        index_in_utterance = idx - self.item_id_shift[item_id] + self.chunk_len // 2
        current_features = self.dataset[item_id][self.feature_field][
            index_in_utterance
            - self.chunk_len // 2 : index_in_utterance
            - self.chunk_len // 2
            + self.chunk_len
        ]

        return {self.feature_field: current_features, "target": self.all_targets[idx]}


class ChunkedDataset(torch.utils.data.Dataset):
    """A dataset that selects fixed-sized chunks of sequential data."""

    def __init__(
        self,
        dataset,
        chunk_len,
        varlen_fields=("features",),
        drop_fields=(),
        training=False,
        transform=None,
        oversample=1,
    ):
        self.dataset = utils.construct_from_kwargs(dataset)
        self.chunk_len = chunk_len
        self.varlen_fields = varlen_fields
        self.drop_fields = set(drop_fields)
        self.training = training
        if transform:
            self.transform = utils.construct_from_kwargs(transform)
        else:
            self.transform = None
        if not training:
            self.transform = None
        self.oversample = oversample
        assert self.training or self.oversample == 1

    def __len__(self):
        return len(self.dataset) * self.oversample

    def __getitem__(self, idx):
        item = self.dataset[idx // self.oversample]
        ret = {}
        ref_len = item[self.varlen_fields[0]].size(0)
        ref_offset = ref_len - self.chunk_len

        assert ref_len > self.chunk_len, f"Item #{idx} is too short ({ref_len} > {self.chunk_len})."

        if self.training:
            ref_offset = random.randrange(ref_offset)
        else:
            ref_offset = ref_offset // 2
        for k, v in item.items():
            if k in self.drop_fields:
                continue
            if k in self.varlen_fields:
                len_ratio = v.size(0) / ref_len
                chunk_len = int(self.chunk_len * len_ratio)
                chunk_offset = int(ref_offset * len_ratio)
                v = v[chunk_offset : chunk_offset + chunk_len]
            ret[k] = v
        if self.transform:
            ret = self.transform(ret)
        return ret

    @property
    def metadata(self):
        return self.dataset.metadata


class SpeechDataset(torch.utils.data.Dataset):
    """A dataset that loads speech utterances from Kaldi matrix files"""

    def __init__(
        self,
        modalities_opts,
        transform=None,
        text_file=None,
        vocabulary_file=None,
        utt2spk_file=None,
        ali_file=None,
        split_by_space=False,
        cmvn_normalize_var=False,
    ):
        self.uttids = {}
        self.features = {}
        self.feature_delta_dim = {}
        self.cmvn = {}
        self.utt_centered = {}

        assert "features" in modalities_opts, f"Currently modalities_opts requires at least 'features' modality."

        for modality, modality_opts in modalities_opts.items():
            feature_file = modality_opts.pop('file')
            feature_delta_dim = modality_opts.pop('feature_delta_dim')
            cmvn_file = modality_opts.pop('cmvn_file')
            utt_centered = modality_opts.pop('utt_centered')

            self.uttids[modality], self.features[modality] = zip(*self._read_scp_file(feature_file))
            feature_dir = os.path.dirname(feature_file)
            self.features[modality] = OrderedDict(
                [
                    (uttid, self._to_absolute_path(feature_dir, f))
                    for uttid, f in self._read_scp_file(feature_file)
                ]
            )
            self.uttids[modality] = list(self.features[modality].keys())

            self.feature_delta_dim[modality] = feature_delta_dim
            self.utt_centered[modality] = utt_centered

            if cmvn_file:
                self.cmvn[modality] = self._load_cmvn(cmvn_file)
            else:
                self.cmvn[modality] = None

            assert not modality_opts, f"modality_opts keys {modality_opts.keys()} not used."

        self._keep_common_uttids()

        self.split_by_space = split_by_space

        self.utt2spk = None
        if utt2spk_file is not None:
            self.utt2spk = dict(self._read_scp_file(utt2spk_file))
            self._restrict_uttids(self.utt2spk, f"utt2spk ({utt2spk_file})")
            self.speakers = list(set([s for u, s in self.utt2spk.items()]))
            self.speakers_to_idx = {s: i for i, s in enumerate(self.speakers)}
            print(len(self.speakers), "speakers found")

        self.cmvn_normalize_var = cmvn_normalize_var

        if text_file:
            self.text = dict(self._read_scp_file(text_file))
            self._restrict_uttids(self.text, f"text ({text_file})")
            assert vocabulary_file is not None
            self.alphabet = self._read_vocabulary_file(vocabulary_file)
            self.text_int = self._tokenize_text(self.text, self.alphabet)
        else:
            self.text = None

        if ali_file:
            self.align = self._read_align_file(ali_file)
            self._restrict_uttids(self.align, f"align ({ali_file})")
        else:
            self.align = None
        super(SpeechDataset, self).__init__()

        self.metadata = {
            "alignment": {"type": "categorical", "num_categories": len(self.alphabet)},
            "targets": {"type": "categorical", "num_categories": len(self.alphabet)},
        }

        if transform:
            self.transform = utils.construct_from_kwargs(transform)
        else:
            self.transform = None

    def _keep_common_uttids(self):
        self.common_uttids = None
        self.all_uttids = set()

        for uttids in self.uttids.values():
            self.common_uttids = set(uttids) if self.common_uttids is None else self.common_uttids & set(uttids)
            self.all_uttids |= set(uttids)

        for modality, uttids in self.uttids.items():
            missing_uttids = self.all_uttids - set(uttids)
            if len(missing_uttids):
                utils.log(f"Missing features of modality {modality}. "
                          f"Ignoring the following utterances in the dataset:",
                          extra_msg=f"{sorted(list(missing_uttids))}",
                          level=logging.WARNING,
                          once=True)

        assert self.common_uttids is not None
        if not len(self.common_uttids):
            utils.log("No common uttids, dataset empty.", level=logging.WARNING,
                      once=True)

    def _restrict_uttids(self, field_dict, field_name):
        # process as a list to keep the ordering
        new_uttids = []
        missing_uttids = []
        for u in self.common_uttids:
            if u in field_dict:
                new_uttids.append(u)
            else:
                missing_uttids.append(u)
        self.common_uttids = new_uttids
        if missing_uttids:
            mod_logger.warning(
                "SpeechDataset asked to include %s which does not "
                "have the following %s uttids %s",
                field_name,
                len(missing_uttids),
                missing_uttids,
            )

    def __len__(self):
        return len(self.common_uttids)

    def __getitem__(self, idx):
        uttid = self.common_uttids[idx]

        result = {"uttid": uttid}

        result.update({modality: torch.tensor(self._load_feature(self.features[modality][uttid],
                                                                 self.cmvn[modality],
                                                                 self.feature_delta_dim[modality],
                                                                 self.utt_centered[modality]))
                       for modality in self.features})

        feature_lengths = {modality: result[modality].size(0) for modality in self.features}

        if len(set(feature_lengths.values())) != 1:
            utils.log(f"The features selected in the SpeechDataset have different lengths.",
                      extra_msg=f"Feature lengths: {feature_lengths}. "
                      f"Using the length of 'features' "
                      f"for the alignments.",
                      level=logging.WARNING,
                      once=True)

        feature_length = feature_lengths['features']

        if self.align:
            alignment = self.align[uttid]
            align = np.zeros(feature_length)
            align_rle = np.zeros((len(alignment), 2))
            for (i, seq) in enumerate(alignment):
                align[seq[1]: seq[2]] = seq[0]
                align_rle[i] = [seq[1], seq[2]]
            result["alignment"] = torch.tensor(align, dtype=torch.int64)
            result["alignment_rle"] = torch.tensor(align_rle, dtype=torch.int64)

            for modality in self.features:
                self._align_features_to_alignment(result, modality)

        if self.text:
            result["targets"] = torch.tensor(self.text_int[uttid])

        if self.utt2spk is not None:
            result["spk"] = self.utt2spk[uttid]
            result["spkid"] = self.speakers_to_idx[result["spk"]]

        if self.transform:
            result = self.transform(result)
        return result

    def _align_features_to_alignment(self, result, field):
        _ = self
        fea_size = result[field].size(0)
        ali_size = result['alignment'].size(0)

        if fea_size == ali_size:
            # Nothing to do here
            return

        elif fea_size < ali_size:
            utils.log('Features are smaller than alignment.',
                      extra_msg=f' Features {field} length {fea_size} != alignment lengths {ali_size}',
                      level=logging.WARNING,
                      once=True)

            result[field] = F.pad(result[field],
                                  pad=[0, 0,
                                       0, 0,
                                       0, ali_size - fea_size])

        else:
            utils.log('Features are larger than alignment.',
                      extra_msg=f' Features {field} length {fea_size} != alignment lengths {ali_size}',
                      level=logging.WARNING,
                      once=True)

            result[field] = result[field][:ali_size]

        return

    def _load_feature(self, feat_path, cmvn, feature_delta_dim, utt_centered):
        mat = kaldi_io.read_mat(feat_path).copy()

        if cmvn is not None:
            mat = self._apply_cmvn(mat, cmvn)

        feat_dim = (-1, feature_delta_dim, mat.shape[1] // feature_delta_dim)
        mat = mat.reshape(feat_dim).transpose((0, 2, 1))

        if utt_centered:
            mat = mat - mat.mean(dim=0)

        return mat

    def _apply_cmvn(self, features, cmvn):
        # https://github.com/kaldi-asr/kaldi/blob/master/src/transform/cmvn.cc
        means, stds = cmvn
        features -= means
        if self.cmvn_normalize_var:
            features /= stds
        return features

    def _load_cmvn(self, cmvn_file):
        cmvn = kaldi_io.read_mat(cmvn_file)
        assert cmvn.shape[0] == 2
        cnt = cmvn[0, -1]
        sums = cmvn[0, :-1]
        sums2 = cmvn[1, :-1]
        means = sums / cnt
        stds = np.sqrt(np.maximum(1e-10, sums2 / cnt - means ** 2))
        return means, stds

    def _to_absolute_path(self, base, path):
        if os.path.isabs(path):
            return path
        else:
            return os.path.join(base, path)

    def _read_scp_file(self, text_file):
        utterances = []
        with open(text_file, "r") as text_f:
            for line in text_f:
                line = line.strip()
                if not line:
                    continue
                utt_id, text = line.split(None, 1)
                utterances.append((utt_id, text))
        return utterances

    def _read_align_file(self, alignment_file):
        align = {}
        with open(alignment_file, "r") as f:
            for line in f:
                line.strip()
                if not line:
                    continue
                utt_id, phones = line.split(None, 1)
                for seq in phones.split(";"):
                    seq = seq.rstrip().lstrip()
                    if seq:
                        (ph, start, end) = seq.split()
                        assert ph in self.alphabet.chars
                        phInt = self.alphabet.ch2idx(ph)
                        if utt_id not in align:
                            align[utt_id] = []
                        align[utt_id].append((phInt, int(start), int(end)))
        return align

    def _read_vocabulary_file(self, vocabulary_file):
        itos = []
        stoi = {}
        with open(vocabulary_file, "r") as f:
            for line in f:
                line = line.strip()
                itos.append(line)
                stoi[line] = len(itos) - 1
        return Alphabet(
            input_dict=stoi,
            translation_dict={" ": "<spc>"},
            unk=("<unk>",),
            blank=("<eps>",),
            space=(),
        )

    def _tokenize_text(self, texts, vocabulary):
        tokenized = {}
        for uttid, text in texts.items():
            if self.split_by_space:
                tokens = text.split(" ")
            else:
                tokens = text
            tokenized[uttid] = np.array([self.alphabet.ch2idx(c) for c in tokens])
        return tokenized


def get_partial_sampler(dataset, ratio, salt=''):
    hashes = [hashlib.md5((salt + str(i)).encode()).digest()
            for i in range(len(dataset))]
    idxs = list(range(len(dataset)))

    idxs.sort(key=lambda i: hashes[i])
    idxs = idxs[:int(len(dataset) * ratio)]

    return torch.utils.data.SubsetRandomSampler(idxs)


class FixedDatasetLoader(torch.utils.data.DataLoader):
    """A DataLoader that collates data of a fixed size.
    """

    def __init__(
        self, dataset, field_names=("features", "targets"), rename_dict=None,
        ratio=None, **kwargs
    ):
        self.field_names = field_names
        self.rename_dict = rename_dict or {}
        dataset = utils.construct_from_kwargs(dataset)
        sampler = None
        if ratio is not None:
            sampler = get_partial_sampler(dataset, ratio)
        super(FixedDatasetLoader, self).__init__(dataset=dataset,
                sampler=sampler, **kwargs)

        if hasattr(self.dataset, "metadata"):
            self.metadata = {
                self.rename_dict.get(k, k): v for k, v in self.dataset.metadata.items()
            }

    def __iter__(self):
        for data in super(FixedDatasetLoader, self).__iter__():
            if isinstance(data, (list, tuple)):
                assert len(data) == len(self.field_names)
                yield dict(zip(self.field_names, data))
            else:
                yield {self.rename_dict.get(k, k): v for k, v in data.items()}


def _combine_field(batch, indicies, field, lengths=None, dynamic_axis=0):
    if not lengths:
        lengths = [example[field].shape[dynamic_axis] for example in batch]
    lengths = np.array(lengths, "int32")
    max_length = np.max(lengths)

    return (
        torch.from_numpy(
            np.stack(
                [
                    np.lib.pad(
                        batch[i][field].numpy(),
                        [
                            (0, 0)
                            if j != dynamic_axis
                            else (0, max_length - lengths[i])
                            for j in range(batch[0][field].numpy().ndim)
                        ],
                        "constant",
                        constant_values=0,
                    )
                    for i in indicies
                ]
            )
        ),
        torch.from_numpy(lengths[indicies]),
    )


class PaddingCollater:
    def __init__(self, varlen_fields, rename_dict):
        self.rename_dict = rename_dict
        self.main_feature = varlen_fields[0]
        self.varlen_fields = set(varlen_fields)

    def __call__(self, batch):
        if self.rename_dict:
            batch = [
                {self.rename_dict.get(k, k): v for k, v in e.items()} for e in batch
            ]
        feat_lengths = [example[self.main_feature].shape[0] for example in batch]
        indices = np.argsort(feat_lengths)[::-1]

        collated_batch = {}
        for k in batch[0]:
            if k in self.varlen_fields:
                v, v_len = _combine_field(batch, indices, k, dynamic_axis=0)
                collated_batch[k] = v
                collated_batch[k + "_len"] = v_len
            else:
                # These are not variable lengths so we will build a list of them
                v = [batch[i][k] for i in indices]

                try:
                    # If they were tensors we will make a torch tensor out of the list
                    v = torch.tensor(v)

                except (ValueError, RuntimeError, TypeError):
                    # We have not been able to make a tensor out of the batch of elements for several possible reasons
                    # - no tensor-like objects
                    # - no shape method
                    # - incompatible shapes
                    utils.log(
                        f"Unable to convert the the batch of elements of field '{k}' into a tensor, "
                        f"leaving as a list.",
                        other_keys=(repr(self),),
                        level=logging.INFO,
                        once=True,
                    )

                collated_batch[k] = v

        return collated_batch


class PaddedDatasetLoader(torch.utils.data.DataLoader):
    """A DataLoader that pads selected fields and sort examples by length.
    The source dataset can be queried for alphabet, alphabet size,
    and vocabulary filename. When training, always write out the
    alphabet for later re-use.
    We expose the alphabet to to trainer/model classes.
    """

    def __init__(
        self,
        dataset,
        varlen_fields=("features", "targets", "alignment"),
        rename_dict=None,
        collate_fn=None,
        ratio=None,
        **kwargs,
    ):
        assert not collate_fn
        if rename_dict is None:
            rename_dict = {}
        dataset = utils.construct_from_kwargs(dataset)
        collate_fn = PaddingCollater(
            varlen_fields=varlen_fields, rename_dict=rename_dict
        )
        sampler = None
        if ratio is not None:
            sampler = get_partial_sampler(dataset, ratio)
        super(PaddedDatasetLoader, self).__init__(
            dataset=dataset, collate_fn=collate_fn, sampler=sampler, **kwargs
        )

        self.metadata = {
            rename_dict.get(k, k): v for k, v in self.dataset.metadata.items()
        }


SegmentInfo = namedtuple("SegmentInfo", ["item_id", "starts", "ends"])


class FramePairDataset(torch.utils.data.Dataset):
    """
    Dataset that iterates over pairs of consequtive frames, where each
    pair originates from the same segment. Depends on DistributionMatchingDataset.
    """

    def __init__(self, distrib_matching_dataset):
        self.distrib_matching_dataset = distrib_matching_dataset
        self.chunk_len = self.distrib_matching_dataset.chunk_len
        self.feature_field = self.distrib_matching_dataset.feature_field

    def __len__(self):
        return len(self.distrib_matching_dataset.frame_pairs)

    def __getitem__(self, idx):
        pair = self.distrib_matching_dataset.frame_pairs[idx]
        item_id = pair[0]
        position = pair[1]
        feature_pairs = torch.stack(
            [
                self.distrib_matching_dataset.dataset[item_id][self.feature_field][
                    position
                    - self.chunk_len // 2 : position
                    - self.chunk_len // 2
                    + self.chunk_len
                ],
                self.distrib_matching_dataset.dataset[item_id][self.feature_field][
                    position
                    - self.chunk_len // 2
                    + 1 : position
                    - self.chunk_len // 2
                    + self.chunk_len
                    + 1
                ],
            ]
        )
        return {
            "feature_pairs": feature_pairs,
            "key": item_id,
            "frame_number": position,
        }


class DistributionMatchingDataset(torch.utils.data.Dataset):
    """
    A dataset for distribution matching, constructed from an underlying sequential dataset.
    Contains sampled chunks from `n=order` consequtive (phone or character) segments.
    Expects that the underlying dataset has fields $feature_field,
    text and alignment_rle
    """

    def __init__(
        self,
        dataset,
        chunk_len,
        order=5,
        lm_probs=None,
        num_ngrams=10000,
        feature_field="image",
    ):
        self.dataset = utils.construct_from_kwargs(dataset)
        self.chunk_len = chunk_len
        self.order = order
        self.feature_field = feature_field
        self.alphabet = self.dataset.alphabet

        # For distribution matching, we need to know the list of output
        # symbols, without the blank.
        # We create a mapping between outputs and alphabet items
        self.alphabet_to_output = {}
        self.outputs = []
        for char, idx in self.dataset.alphabet.chars.items():
            if (
                char not in self.dataset.alphabet.blank
                and char != self.dataset.alphabet.blank
            ):
                self.outputs.append(idx)
                self.alphabet_to_output[idx] = len(self.alphabet_to_output)

        self.output_frequencies = torch.zeros(len(self.outputs))

        #  if lm_probs is given, read it from there
        if lm_probs is not None:
            logging.info(f"Reading LM n-gram probabilities from {lm_probs}")

            self.lm_probs = []
            self.lm_probs_ngrams = []
            for l in open(lm_probs):
                ss = l.split()
                assert self.order == len(ss) - 1
                lm_prob = float(ss[0])
                self.lm_probs.append(lm_prob)
                self.lm_probs_ngrams.append(
                    [self.alphabet_to_output[self.alphabet.ch2idx(p)] for p in ss[1:]]
                )

                for p in ss[1:]:
                    self.output_frequencies[
                        self.alphabet_to_output[self.alphabet.ch2idx(p)]
                    ] += 1
            self.output_frequencies /= sum(self.output_frequencies)
            self.lm_probs = torch.tensor(np.array(self.lm_probs)).float()
            self.lm_probs_ngrams = torch.tensor(np.array(self.lm_probs_ngrams))

        else:
            self.lm_probs = None
            ngram_counter = Counter()

        self.segments = []
        self.frame_pairs = []
        logging.info("Iterating over the dataset to count and index segments")
        for i, item in enumerate(self.dataset):
            ali_rle = item["alignment_rle"]
            # we take overlapping segments
            for j in range(len(ali_rle) - order + 1):
                self.segments.append(
                    SegmentInfo(
                        i,
                        torch.tensor([int(a[0]) for a in ali_rle[j : j + order]]),
                        torch.tensor([int(a[1]) for a in ali_rle[j : j + order]]),
                    )
                )

            for j in range(len(ali_rle)):
                for k in range(
                    max(self.chunk_len // 2, int(ali_rle[j, 0])),
                    min(
                        len(item[self.feature_field]) - self.chunk_len // 2 - 1,
                        int(ali_rle[j, 1]),
                    ),
                ):
                    self.frame_pairs.append((i, k))

            # If LM probs are not given from outside, we use the 'text' field of the underlying dataset
            # and count the n-grams
            # FIXME: use a trie, if the amount of n-grams gets large
            if self.lm_probs is None:
                ngrams = zip(
                    *[
                        [self.alphabet_to_output[idx.item()] for idx in item["text"]][
                            i:
                        ]
                        for i in range(self.order)
                    ]
                )
                ngram_counter.update(ngrams)

                for output in [
                    self.alphabet_to_output[idx.item()] for idx in item["text"]
                ]:
                    self.output_frequencies[output] += 1

        if lm_probs is None:
            logging.info("Computing LM n-gram probabilities from reference text")
            self.lm_probs = []
            self.lm_probs_ngrams = []
            num_total_ngrams = sum(ngram_counter.values())
            for ngram, count in ngram_counter.most_common(num_ngrams):
                self.lm_probs.append(count / num_total_ngrams)
                self.lm_probs_ngrams.append(ngram)

            self.lm_probs = torch.tensor(np.array(self.lm_probs)).float()
            self.lm_probs_ngrams = torch.tensor(np.array(self.lm_probs_ngrams))

            self.output_frequencies /= sum(self.output_frequencies)

        if Globals.cuda:
            self.lm_probs = self.lm_probs.to("cuda")
            self.lm_probs_ngrams = self.lm_probs_ngrams.to("cuda")

        self.stddev = np.std(
            np.array(
                [
                    segment.ends[i] - segment.starts[i] + 1
                    for segment in self.segments
                    for i in range(len(segment.starts))
                ]
            )
        )

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment = self.segments[idx]
        item_id = segment.item_id
        starts = segment.starts
        ends = segment.ends
        sampled_positions = utils.sample_truncated_normal(starts, ends, self.stddev)

        if self.chunk_len > 1:
            sampled_positions = torch.clamp(
                sampled_positions,
                self.chunk_len // 2,
                len(
                    self.dataset[item_id][self.feature_field] - self.chunk_len // 2 - 1
                ),
            )
            sampled_features = torch.stack(
                [
                    self.dataset[item_id][self.feature_field][
                        sampled_position
                        - self.chunk_len // 2 : sampled_position
                        - self.chunk_len // 2
                        + self.chunk_len
                    ]
                    for sampled_position in sampled_positions.tolist()
                ]
            )
        else:
            # This makes it about 30% faster in cases where chunk_len=1
            # (like speech features with splicing already applied)
            sampled_features = self.dataset[item_id][self.feature_field][
                sampled_positions
            ].unsqueeze(1)

        return {
            "sampled_frames": sampled_features,
            "key": item_id,
            "starts": starts,
            "ends": ends,
            "sampled_positions": sampled_positions,
        }

    def get_frame_pair_iter(self, batch_size):
        """
        Returns an iterator that loops infinitely over random pairs of consequtive frames
        that originate from the same segment.
        """

        def looping_iter(dl):
            while True:
                for x in iter(dl):
                    yield x

        frame_pair_dataset = FramePairDataset(self)
        dataloader = torch.utils.data.DataLoader(
            frame_pair_dataset, batch_size=batch_size, shuffle=True
        )
        return looping_iter(dataloader)


class ImageDataset(torch.utils.data.TensorDataset):
    def __init__(self, path, split):
        if split == 'train':
            dset = datasets.SVHN(path, split='train', download=True)
            feats = (dset.data.astype(np.float32) / 255.0 - 0.5) * 2.0
            labels = dset.labels
        elif split == 'dev':
            dset = datasets.SVHN(path, split='train', download=True)
            feats = (dset.data[:10000].astype(np.float32) / 255.0 - 0.5) * 2.0
            labels = dset.labels[:10000]
        elif split == 'test':
            print('Note: 10k test samples set aside as dev')
            dset = datasets.SVHN(path, split='test', download=True)
            feats = (dset.data[10000:].astype(np.float32) / 255.0 - 0.5) * 2.0
            labels = dset.labels[10000:]
        else:
            raise ValueError('Unknown dataset split')
        self.num_classes = len(np.unique(labels))
        super(SVHNDataset, self).__init__(torch.from_numpy(feats),
                                          torch.from_numpy(labels))


class ImageDataset(torch.utils.data.Dataset):
    """A dataset that loads static (non-temporal) images"""

    def __init__(
        self,
        modalities_opts,
        transform=None,
        text_file=None,
        vocabulary_file=None,
        utt2spk_file=None,
        ali_file=None,
        split_by_space=False,
        cmvn_normalize_var=False,
    ):
        self.uttids = {}
        self.features = {}
        self.feature_delta_dim = {}
        self.cmvn = {}
        self.utt_centered = {}

        assert "features" in modalities_opts, f"Currently modalities_opts requires at least 'features' modality."

        if transform:
            self.transform = utils.construct_from_kwargs(transform)
        else:
            self.transform = None

    def __len__(self):
        return len(self.common_uttids)

    def __getitem__(self, idx):
        uttid = self.common_uttids[idx]

        result = {"uttid": uttid}

        result.update({modality: torch.tensor(self._load_feature(self.features[modality][uttid],
                                                                 self.cmvn[modality],
                                                                 self.feature_delta_dim[modality],
                                                                 self.utt_centered[modality]))
                       for modality in self.features})

        feature_lengths = {modality: result[modality].size(0) for modality in self.features}

        if len(set(feature_lengths.values())) != 1:
            utils.log(f"The features selected in the SpeechDataset have different lengths.",
                      extra_msg=f"Feature lengths: {feature_lengths}. "
                      f"Using the length of 'features' "
                      f"for the alignments.",
                      level=logging.WARNING,
                      once=True)

        feature_length = feature_lengths['features']

        if self.align:
            alignment = self.align[uttid]
            align = np.zeros(feature_length)
            align_rle = np.zeros((len(alignment), 2))
            for (i, seq) in enumerate(alignment):
                align[seq[1]: seq[2]] = seq[0]
                align_rle[i] = [seq[1], seq[2]]
            result["alignment"] = torch.tensor(align, dtype=torch.int64)
            result["alignment_rle"] = torch.tensor(align_rle, dtype=torch.int64)

            for modality in self.features:
                self._align_features_to_alignment(result, modality)

        if self.text:
            result["targets"] = torch.tensor(self.text_int[uttid])

        if self.utt2spk is not None:
            result["spk"] = self.utt2spk[uttid]
            result["spkid"] = self.speakers_to_idx[result["spk"]]

        if self.transform:
            result = self.transform(result)
        return result
