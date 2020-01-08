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

from distsup.logger import DefaultTensorLogger
from distsup.models.vqvae import vqvae_nanxin
from distsup.trainer import Trainer

logger = DefaultTensorLogger()


def main():
    from distsup.data import PaddedDatasetLoader

    trainer = Trainer(num_epochs=1, learning_rate=0.004, optimizer_name='Adam')
    dataset = PaddedDatasetLoader({'class_name': 'egs.mnist_seq.data.MNISTSequentialDataset',
                                   'target_height': 28,
                                   'max_samples': 10,
                                   'split': 'train'},
                                  rename_dict={'image': 'features',
                                               'text': 'targets'},
                                  varlen_fields=['features', 'targets', 'alignment'],
                                  batch_size=32)

    eval_dataset = PaddedDatasetLoader({'class_name': 'egs.mnist_seq.data.MNISTSequentialDataset',
                                        'target_height': 28,
                                        'max_samples': 5,
                                        'split': 'test'},
                                       rename_dict={'image': 'features',
                                                    'text': 'targets'},
                                       varlen_fields=['features', 'targets', 'alignment'],
                                       batch_size=32)

    model = vqvae_nanxin(dataloader=dataset)
    # print(model.evaluate(dataset))

    trainer.run('/tmp/test_distsup',
                model,
                dataset,
                eval_datasets={'eval': eval_dataset},
                debug_skip_training=True)


if __name__ == '__main__':
    main()
