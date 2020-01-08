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

import os

from distsup.data import SpeechDataset


VOCAB_DIR = os.path.abspath(os.path.dirname(__file__))


WSJ_BASE_DIR = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'data', 'wsj'))

WSJ_PASE_BASE_DIR = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'data', 'wsj_pase'))


class WSJData(SpeechDataset):
    def __init__(self, split, text_fname='text', vocab_fname='vocabulary.txt',
                 ali_file=None, split_by_space=False,
                 modalities={'features': 'fbanks'},
                 **kwargs):
        modality_opts = {
            'fbanks': dict(file=os.path.join(WSJ_BASE_DIR, split, 'feats.scp'),
                           cmvn_file=os.path.join(WSJ_BASE_DIR, 'train_si284', 'cmvn'),
                           feature_delta_dim=3,
                           utt_centered=False),
            'mfccs': dict(file=os.path.join(WSJ_BASE_DIR, split, 'mfccs.scp'),
                          cmvn_file=os.path.join(WSJ_BASE_DIR, 'train_si284', 'mfccs_cmvn'),
                          feature_delta_dim=3,
                          utt_centered=False),
            'wavs': dict(file=os.path.join(WSJ_BASE_DIR, split, 'wav_as_feats.scp'),
                         cmvn_file=None,
                         feature_delta_dim=1,
                         utt_centered=False),
            'pase': dict(file=os.path.join(WSJ_PASE_BASE_DIR, split, 'pase.scp'),
                         cmvn_file=None,
                         feature_delta_dim=1,
                         utt_centered=False),
            'pase-uttcentered': dict(file=os.path.join(WSJ_PASE_BASE_DIR, split, 'pase.scp'),
                                     cmvn_file=None,
                                     feature_delta_dim=1,
                                     centered=True),
        }

        super(WSJData, self).__init__(
            modalities_opts={k: modality_opts[v] for k, v in modalities.items()},
            text_file=os.path.join(WSJ_BASE_DIR, split, text_fname),
            ali_file=os.path.join(WSJ_BASE_DIR, split, ali_file) if ali_file else None,
            vocabulary_file=os.path.join(WSJ_BASE_DIR, 'train_si284', vocab_fname),
            utt2spk_file=os.path.join(WSJ_BASE_DIR, split, 'utt2spk'),
            split_by_space=split_by_space,
            **kwargs
        )
        self.num_classes = len(self.alphabet)
