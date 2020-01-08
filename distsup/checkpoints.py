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
import glob
import logging
import os
import sys

import torch

from distsup import configuration, utils


mod_logger = logging.getLogger(__name__)


def load_state(path):
    if configuration.Globals.cuda:
        map_location = None
    else:
        map_location = 'cpu'
    return torch.load(path, map_location=map_location)


def glob_latest(pattern):
    pl, pr = pattern.split('*')
    files = glob.glob(pattern)
    if not files:
        return None

    def key(x):
        return int(x[len(pl):-len(pr)])
    return max(files, key=key)


def latest_checkpoint(save_dir):
    return glob_latest(os.path.join(
        save_dir, 'checkpoints', 'checkpoint_*.pkl'))


def latest_config(save_dir):
    return glob_latest(os.path.join(
        save_dir, 'train_config*.yaml'))


def get_config_and_model(model_or_dir, config=None, polyak=None,
                         no_strict=False, modify_dict=None,
                         cuda=torch.cuda.is_available()):
    """Loads a pretrained model.
    """
    configuration.Globals.cuda = cuda
    if modify_dict is None:
        modify_dict = {}
    model = None

    if os.path.isdir(model_or_dir):
        rundir = model_or_dir
    else:
        if model_or_dir.lower().endswith('.yaml'):
            assert not config, ("Can't point the model to a yaml file and "
                                "specify a config at the same time")
            config = model_or_dir
            rundir = os.path.dirname(config)
        else:
            assert (model_or_dir.lower().endswith('.pkl') and
                    "checkpoints" in model_or_dir.lower()), (
                    "The model should point to a pickle file under "
                    "rundir/checkpints")
            model = model_or_dir
            rundir = os.path.dirname(os.path.dirname(model_or_dir))

    config = config or latest_config(rundir)
    model = model or latest_checkpoint(rundir)

    mod_logger.info("Using model dir %s", rundir)
    mod_logger.info("Loading config from %s", config)
    config = configuration.Configuration(config, modify_dict)

    mod_logger.info("Loading saved model from %s", model)
    state_dict = load_state(model)
    state_dict_name = 'state_dict'
    if polyak:
        if polyak == 'AUTO':
            state_dict_name = max(
                [k for k in state_dict if k.startswith('avg_state_dict')],
                key=utils.natural_keys)
        else:
            state_dict_name = 'avg_state_dict_%f' % (float(polyak),)
    mod_logger.info("Loading state dict %s", state_dict_name)
    model = config['Model']
    model.load_state_dict(state_dict[state_dict_name], strict=(not no_strict))

    configuration.Globals.epoch = state_dict['epoch']
    configuration.Globals.current_iteration = state_dict['current_iteration']

    if configuration.Globals.cuda:
        model.cuda()
    model.eval()

    return config, model


def get_common_model_loading_argparser():
    """Ger an argparser set with options common to loading trau=ined models.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("model_or_dir",
                        help="Path to the model or the run directory. The "
                             "function also accepts here a specific Yaml from "
                             "the run dir.")
    parser.add_argument("-c", "--config",
                        nargs="?", help="Path to the config file")
    parser.add_argument("-p", "--polyak",
                        help="Use Polyak averaged model, using the string "
                             "'AUTO' will load the first Polyak dict found",
                        default=None, type=str)
    parser.add_argument('-m', '--modify_config', nargs='+', action='append',
                        help="List of config modifications")
    parser.add_argument('--cuda', default=torch.cuda.is_available(),
                        help='use CUDA', type=utils.str2bool)
    parser.add_argument('--no-strict', action='store_true',
                        help="allow unknown params in pickles")
    return parser


def get_config_and_model_from_argparse_args(args):
    """Load a model using argparser from `get_common_model_loading_argparser`.
    """
    modify_dict = utils.extract_modify_dict(args.modify_config)
    return get_config_and_model(
        model_or_dir=args.model_or_dir, config=args.config, polyak=args.polyak,
        no_strict=args.no_strict, modify_dict=modify_dict,
        cuda=args.cuda)


class Checkpointer(object):
    '''
    The checkpointer. Stores:
        * last_n checkpoints (with filenames checkpoint_STEPNUM.pkl)
        * one checkpoint every n_hours (with filenames: checkpoint_STEPNUM.pkl)
        * one best checkpoint for every logged channel
          (filename: best_STEPNUM_CHANNELNAME_VALUE.pkl)
    '''
    def __init__(self, last_n=3, every_n_hours=1, enabled=True):
        self.last_n = last_n
        self.every_n_seconds = int(60 * 60 * every_n_hours)
        self.last_reload_iteration = None
        self.checkpoints = []
        self.best_checkpoints = {}
        self.enabled = enabled

    def set_save_dir(self, save_dir):
        self.save_dir = save_dir

    def load_checkpointer_state(self, iteration):
        if not self.enabled:
            return
        if iteration == self.last_reload_iteration:
            return
        filenames = [f[:-4] for f in os.listdir(self.create_path())]
        self.checkpoints = \
            [(f, int(os.path.getmtime(self.create_path(f))))
             for f in filenames if f.startswith('checkpoint_')]
        self.checkpoints = sorted(self.checkpoints,
                                  key=lambda x: int(x[0][x[0].rindex('_')+1:]))

        self.best_checkpoints = {}
        best_checkpoints = [f for f in filenames if f.startswith('best_')]
        for f in best_checkpoints:
            fields = f.split('_')
            cerval = float(fields[3])
            chann = fields[2]
            self.best_checkpoints[chann] = (cerval, f)
        self.last_reload_iteration = iteration

    def remove_unnecessary_checkpoints(self):
        if not self.checkpoints:
            return
        last_safe = self.checkpoints[0]
        removeable_checkpoints = []
        for checkpoint in self.checkpoints[1:]:
            if checkpoint[1] - last_safe[1] >= self.every_n_seconds:
                last_safe = checkpoint
            else:
                removeable_checkpoints += [checkpoint]
        for rem in removeable_checkpoints[:(-1*self.last_n)]:
            self.remove(rem[0])

    def create_path(self, filename=None):
        assert self.save_dir, "Checkpointer: Save dir not set"
        if filename is None:
            return os.path.join(self.save_dir, 'checkpoints')
        else:
            return os.path.join(self.save_dir,
                                'checkpoints', '{}.pkl'.format(filename))

    def save(self, filename, current_iteration, epoch, model, optimizer,
             lr_scheduler):
        print("  + saving {}".format(filename))

        possible_filename = 'checkpoint_{}'.format(current_iteration)
        oname = self.create_path(filename)
        if any((k[0] == possible_filename for k in self.checkpoints)):
            print("  + iteration {} already saved, making hard link"
                  .format(current_iteration))
            os.link(self.create_path(possible_filename), oname)
            return
        else:
            temp_path = self.create_path('.{}.pkl.temporary'.format(filename))
            state_dict = {
                'current_iteration': current_iteration,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }
            for k in model.__dict__:
                if k.startswith('avg_state_dict'):
                    print("  + saving polyak", k)
                    state_dict[k] = getattr(model, k)
            utils.ensure_dir(os.path.dirname(oname))
            torch.save(state_dict, temp_path)
            os.rename(temp_path, oname)

    def remove(self, filename):
        print("  + removing {}".format(filename))
        oname = self.create_path(filename)
        os.remove(oname)

    def try_checkpoint_best(self, name, value,
                            iteration_num, epoch_num,
                            model, optimizer, lr_scheduler):
        if not self.enabled:
            return
        self.load_checkpointer_state(iteration_num)
        if name not in self.best_checkpoints \
           or value < self.best_checkpoints[name][0]:
            point_name = 'best_{}_{}_{}'.format(iteration_num, name, value)
            self.save(point_name,
                      iteration_num, epoch_num,
                      model, optimizer, lr_scheduler)
            if name in self.best_checkpoints:
                self.remove(self.best_checkpoints[name][1])
            self.best_checkpoints[name] = (value, point_name)

    def checkpoint(self, interation_num, epoch_num,
                   model, optimizer, lr_scheduler):
        if not self.enabled:
            return
        point_name = 'checkpoint_{}'.format(interation_num)
        self.save(point_name,
                  interation_num, epoch_num,
                  model, optimizer, lr_scheduler)
        self.load_checkpointer_state(interation_num)
        self.remove_unnecessary_checkpoints()
