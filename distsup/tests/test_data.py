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

import unittest

import torch.utils.data

from distsup import utils


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super(DummyDataset, self).__init__(**kwargs)
        self.items = [{'num': i, 'str': str(i)} for i in range(10)]
        self.counts = [0] * len(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        self.counts[i] += 1
        return self.items[i]

    def get_counts(self, i):
        return self.counts[i]

    @property
    def count_prop(self):
        return self.counts


class TestCachedDataset(unittest.TestCase):
    def test_all(self):
        conf = {
            'class_name': 'distsup.data.CachedDataset',
            'real_class_name': 'distsup.tests.test_data.DummyDataset',
        }
        ds = utils.construct_from_kwargs(conf)
        assert ds[0]['num'] == 0
        assert len(ds) == len(ds._wrapped) == 10
        for i in range(3):
            ds[1]
        assert ds.counts[0] == ds.counts[1] == 1
        assert ds.counts[2] == ds.counts[3] == 0
        assert ds.get_counts(0) == 1
        assert ds.count_prop[0] == 1


if __name__ == '__main__':
    unittest.main()
