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
import copy
import functools
import logging
import os
import sys
import uuid

import torch

from distsup.configuration import Configuration, Globals
from distsup.utils import extract_modify_dict, str2bool
from distsup.checkpoints import latest_checkpoint, load_state


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to the config file")
    parser.add_argument('save_dir', help="Directory to save all model files")
    parser.add_argument('-c', '--continue-training',
                        help='Continue experiment from given checkpoint')
    parser.add_argument('-m', '--modify_config', nargs='+', action='append',
                        help="List of config modifications")
    parser.add_argument('--cuda', default=torch.cuda.is_available(),
                        help='Use CUDA', type=str2bool)
    parser.add_argument('--rng_seed', default=None, type=int,
                        help='Reset the rng seed')
    parser.add_argument('--initialize-from', default=None,
                        help='Load weights from')
    parser.add_argument('--debug-anomaly', action='store_true',
                        help='For debugging nan or inf in computation graph')
    parser.add_argument('-d', '--debug-skip-training', action='store_true',
                        help='For debugging finish training after 1 mnibatch')
    parser.add_argument('--remote-log', action='store_true',
                        help='Send training logs to a remote server')
    parser.add_argument('--cluster', default=os.environ.get('DISTSUP_CLUSTER', ''),
                        help='Cluster name metadata')
    parser.add_argument('-t', '--tag', default=os.environ.get('DISTSUP_EXP_TAG', None),
                        help='Experiment group name metadata')
    return parser


def get_config_filename(save_dir):
    template = 'train_config{}.yaml'
    if os.path.isfile(os.path.join(save_dir, template.format(''))):
        return os.path.join(save_dir, template.format(''))
    else:
        i = 1
        while os.path.isfile(os.path.join(save_dir, template.format(i))):
            i += 1
        return os.path.join(save_dir, template.format(i))


def get_uuid(save_dir):
    fpath = os.path.join(save_dir, 'exp_uuid')
    if os.path.isfile(fpath):
        uuid_ = [l.strip() for l in open(fpath)][0]
    else:
        uuid_ = str(uuid.uuid4())
        with open(fpath, 'w') as f:
            f.write(uuid_)
    return uuid_


def ensure_logger_environ():
    if not 'GOOGLE_BIGQUERY_DATASET' in os.environ:
        raise ValueError('GOOGLE_BIGQUERY_DATASET not set')
    if not 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        raise ValueError('GOOGLE_APPLICATION_CREDENTIALS not set')


def initialize_from(model, path):
    state_dict = load_state(path)['state_dict']
    model_dict = model.state_dict()

    logging.info("Initializing parameters from {}:".format(path))
    loaded = []
    for k in sorted(model_dict.keys()):
        if k in state_dict:
            param = state_dict[k]
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if param.shape != model_dict[k].shape:
                logging.info("Skipping {} due to shappe difference".format(k))
                logging.info("k shape in model: {}".format(model_dict[k].shape))
                logging.info("k shape in checkpoint: {}".format(param.shape))
                continue
            model_dict[k].copy_(param)
            loaded.append(k)
    logging.info("Loaded: {}".format(loaded))
    logging.info("Missing: {}".format(
        sorted(set(model_dict.keys()) - set(state_dict.keys()))))
    logging.info("Unknown: {}".format(
        sorted(set(state_dict.keys()) - set(model_dict.keys()))))


def main():
    parser = get_parser()
    args = parser.parse_args()
    # Grab the GPU as a workaround for clusters that don't enforce GPU alloc.
    if args.cuda:
        import torch
        torch.zeros((1, 1), device='cuda')
    if args.debug_anomaly:
        torch.autograd.set_detect_anomaly(True)

    logging.basicConfig(level=logging.INFO)

    if args.rng_seed is not None:
        logging.info("Reseting the random seed")
        torch.manual_seed(args.rng_seed)

    Globals.cuda = args.cuda
    Globals.cluster = args.cluster
    Globals.exp_tag = args.tag
    Globals.exp_config_fpath = args.config
    Globals.save_dir = args.save_dir
    if args.debug_skip_training or not args.remote_log:
        Globals.remote_log = False
    else:
        Globals.remote_log = True
        ensure_logger_environ()

    modify_dict = extract_modify_dict(args.modify_config)
    config = Configuration(args.config, modify_dict,
                           get_config_filename(args.save_dir))
    Globals.objects_config = copy.deepcopy(config.objects_config)
    Globals.exp_uuid = get_uuid(args.save_dir)

    train_data = config['Datasets']['train']
    eval_data = {key: config['Datasets'][key]
                 for key in config['Datasets'].keys() if key != 'train' and key[:6] != 'probe_'}
    if 'probe_train' in config['Datasets'].keys():
        probe_train_data = config['Datasets']['probe_train']
    model = config['Model']

    if args.initialize_from:
        initialize_from(model, args.initialize_from)

    logging.info("Model summary:\n%s" % (model,))
    logging.info("Model params:\n%s" % ("\n".join(["%s: %s" % (p[0], p[1].size())
                                                   for p in model.named_parameters()])))
    logging.info(f'Experiment UUID: {Globals.exp_uuid}')
    logging.info(f'BigQuery {"enabled" if Globals.remote_log else "disabled"}')
    logging.info("Start training")
    trainer = config['Trainer']

    saved_state = None
    if args.continue_training == 'LAST':
        args.continue_training = latest_checkpoint(args.save_dir)
    if args.continue_training is not None:
        logging.info('Loading state from %s...', args.continue_training)
        saved_state = load_state(args.continue_training)

    if 'probe_train' in config['Datasets'].keys():
        trainer.run(args.save_dir, model, train_data, eval_data,
                saved_state=saved_state,
                debug_skip_training=args.debug_skip_training,
                probe_train_data = probe_train_data)
    else:
        trainer.run(args.save_dir, model, train_data, eval_data,
                saved_state=saved_state,
                debug_skip_training=args.debug_skip_training)


if __name__ == "__main__":
    sys.stderr.write("%s %s\n" % (os.path.basename(__file__), sys.argv))
    main()
