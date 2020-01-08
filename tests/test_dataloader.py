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

import logging

from distsup.data import PaddedDatasetLoader


def main():
    loader = PaddedDatasetLoader({'class_name': 'egs.mnist_seq.data.MNISTSequentialDataset',
                                  'root': 'data/mnist.tight0.125.rand0.100k.10k.zip',
                                  'target_width': None,
                                  'target_height': 28
                                  },
                                 varlen_fields=['image', 'text', 'alignment'],
                                 batch_size=3)
    _ = next(iter(loader))
    _ = next(iter(loader))
    sample_batch = next(iter(loader))
    print(sample_batch)

    logging.basicConfig(level=logging.DEBUG)
    loader = PaddedDatasetLoader({'class_name': 'egs.scribblelens.data.ScribbleLensDataset',
                                  'root': 'data/scribblelens.corpus.v1.zip',
                                  'target_width': 400,
                                  'target_height': 100
                                  },
                                 varlen_fields=['image', 'text'],
                                 batch_size=3)
    _ = next(iter(loader))
    _ = next(iter(loader))
    sample_batch = next(iter(loader))
    print(sample_batch)


if __name__ == '__main__':
    main()
