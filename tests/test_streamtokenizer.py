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

from distsup.models.streamtokenizer import StreamTokenizerNet
import distsup.utils


def main():
    import torch
    import pprint

    # Make a dummy Net to test the evaluation
    class TestStreamTokenizerNet(StreamTokenizerNet):
        def __init__(self,
                     error_rate=0.0,
                     **kwargs):
            super(TestStreamTokenizerNet, self).__init__(**kwargs)

            self.error_rate = error_rate
            self.token_rate = 1

            self.add_probes()

        def align_tokens_to_features(self, batch, tokens):
            if self.token_rate == 1:
                return tokens

            else:
                raise NotImplementedError

        def minibatch_loss_and_tokens(self, batch):
            # The bottlenecks return B x W x 1 x 1
            tokens = batch['alignment'].clone().detach().unsqueeze(dim=-1).unsqueeze(dim=-1)

            if self.token_rate != 1:
                raise NotImplementedError

            mask = distsup.utils.get_mask1d(batch['alignment_len'])
            tokens[torch.bernoulli(mask * self.error_rate) > 0.5] = 100

            return 0., {'err': 0.}, tokens

    from distsup.data import PaddedDatasetLoader

    dataset = PaddedDatasetLoader({'class_name': 'egs.mnist_seq.data.MNISTSequentialDataset',
                                   'target_height': 28,
                                   'split': 'test'},
                                  varlen_fields=['features', 'text', 'alignment'],
                                  rename_dict={'image': 'features'},
                                  batch_size=100)

    model = TestStreamTokenizerNet(error_rate=1.0, dataloader=dataset)
    pprint.pprint(model.evaluate(dataset))

    # trainer.run('/tmp/test_distsup', model, dataset)


if __name__ == '__main__':
    main()
