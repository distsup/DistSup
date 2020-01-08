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

'''
Load and apply a pretrained model and run its evalaution function.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import time

import torch

from distsup import checkpoints
from distsup import configuration


def get_parser():
    parser = checkpoints.get_common_model_loading_argparser()
    parser.add_argument("--subset", help="Which subset to use", default="dev")
    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    parser = get_parser()
    args = parser.parse_args()
    config, model = checkpoints.get_config_and_model_from_argparse_args(args)

    dataset = config['Datasets'][args.subset]

    def iter_batches(dataset):
        tic = 0
        for batch_i, batch in enumerate(dataset):
            if time.time() - tic > 30:
                num_examples = len(next(iter(batch.values())))
                print('Processing batch {}/{} ({} elements)'.format(
                      batch_i, len(dataset), num_examples))
                tic = time.time()
            if configuration.Globals.cuda:
                batch = model.batch_to_device(batch, 'cuda')
            yield batch

    results = model.evaluate(iter_batches(dataset))
    print(results)


if __name__ == "__main__":
    sys.stderr.write("%s %s\n" % (os.path.basename(__file__), sys.argv))
    with torch.no_grad():
        main()
