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

import collections
import logging
from copy import deepcopy
from datetime import datetime

import time

import numpy as np
import pickle

import torch
from torch.nn.utils import clip_grad_norm_

from distsup import checkpoints, utils
from distsup.configuration import Globals
from distsup.logger import DefaultTensorLogger
from distsup.utils import DebugStats
from distsup import summary


logger = DefaultTensorLogger()

DEFAULT_LR_SCHEDULER = {
        'class_name': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'factor': 0.5,
        'patience': 3
        }

INF = float('inf')
NEGINF = float('-inf')


class GradientClipper(object):
    def __init__(self, clip_norm, skip_step_norm=np.inf, **kwargs):
        self.clip_norm = clip_norm
        self.skip_step_norm = skip_step_norm
        self.gstats = None
        super(GradientClipper, self).__init__(**kwargs)

    def clip(self, model):
        unclipped_norm = clip_grad_norm_(
            model.get_parameters_for_optimizer(), self.clip_norm)
        clipped = int(unclipped_norm > self.clip_norm)
        skipped = int(unclipped_norm > self.skip_step_norm)

        if self.gstats is None:
            self.gstats = (
                1,
                unclipped_norm, unclipped_norm, unclipped_norm,
                clipped, skipped)
        else:
            (acc_steps, g_min, g_sum, g_max, n_clipped, n_skipped
             ) = self.gstats
            self.gstats = (
                acc_steps + 1,
                min(g_min, unclipped_norm),
                g_sum + unclipped_norm,
                max(g_max, unclipped_norm),
                n_clipped + clipped,
                n_skipped + skipped)
        if logger.is_currently_logging():
            (acc_steps, g_min, g_sum, g_max, n_clipped, n_skipped
             ) = self.gstats
            logger.log_scalar("gclip/min", g_min)
            logger.log_scalar("gclip/max", g_max)
            logger.log_scalar("gclip/mean", 1.0 * g_sum / acc_steps)
            logger.log_scalar("gclip/clipfrac", 1.0 * n_clipped / acc_steps)
            logger.log_scalar("gclip/skipfrac", 1.0 * n_skipped / acc_steps)
            self.gstats = None
        if clipped:
            print("Grad clipped by ", 1.0 * self.clip_norm / unclipped_norm)
        # Tell the trainer if we should skip this step
        return bool(skipped)


class Progress(object):
    def __init__(self):
        self.start_time = datetime.now()
        self.bend = time.time()
        self.reset()

    def reset(self):
        self.tik = time.time()
        self.steps = 0
        self.loss = 0.0
        self.data_load_time = 0.0
        self.stats_acc = collections.defaultdict(lambda: 0)

    def announce_batch(self, iter, epoch, batch, data_len):
        self.bstart = time.time()
        self.data_load_time += self.bstart - self.bend
        self.epoch = epoch
        self.batch = batch
        self.data_len = data_len
        self.iter = iter

    def update(self, lr, loss, stats=None, flush=False):
        self.bend = time.time()
        self.loss += loss.item()
        self.steps += 1
        if stats:
            for k, v in stats.items():
                is_scalar, scalar_value = utils.maybe_get_scalar(v)
                if is_scalar:
                    self.stats_acc[k] += scalar_value
        if not flush:
            return
        tik_tok = self.bend - self.tik
        for k in self.stats_acc:
            self.stats_acc[k] /= self.steps

        print('It. {: >6} | Ep. {: >3} | Batch {: >5}/{} | '
              'Loss {: >8.2f} | '
              '{: >.2f} s/batch | DataLoad {: >.5f}s | '
              'Lr {: >7.5f} | Epoch {} s | Stats {}'.format(
                self.iter, self.epoch, self.batch, self.data_len,
                self.loss / self.steps,
                tik_tok / self.steps,
                self.data_load_time / self.steps, lr,
                str(datetime.now() - self.start_time).split('.')[0],
                ', '.join(["{}: {:.5f}".format(k, v)
                           for k, v in self.stats_acc.items()
                           if utils.is_scalar(v)])))
        self.reset()


class Trainer(object):
    def __init__(self, num_epochs, learning_rate,
                 optimizer_name, optimizer_kwargs={},
                 learning_rate_scheduler=DEFAULT_LR_SCHEDULER,
                 checkpointer={},
                 checkpoint_frequency_within_epoch=None,
                 weight_noise=0, weight_noise_start_iteration=15000,
                 weight_noise_linear_increase=True,
                 weight_noise_prevent_lr_step=True,
                 log_frequency=100, output_frequency=100, gradient_noise=None,
                 gradient_clipping=None,
                 kill_on_nan=True,
                 log_layers_stats=False, polyak_decay=0,
                 codebook_lr=None):
        super(Trainer, self).__init__()

        self.log_frequency = log_frequency
        self.output_frequency = output_frequency
        self.num_epochs = num_epochs
        self.weight_noise = weight_noise
        self.weight_noise_start_iteration = weight_noise_start_iteration
        self.weight_noise_linear_increase = weight_noise_linear_increase
        self.weight_noise_prevent_lr_step = weight_noise_prevent_lr_step
        self.learning_rate = learning_rate
        self.codebook_lr = codebook_lr
        self.lr_scheduler_params = learning_rate_scheduler
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = optimizer_kwargs
        self.gradient_clipper = None
        if gradient_clipping:
            self.gradient_clipper = GradientClipper(**gradient_clipping)

        self.best_values = {}
        self.current_iteration = 0
        self.log_layers_stats = log_layers_stats
        self.kill_on_nan = kill_on_nan
        self.polyak_decay = polyak_decay
        self.gradient_noise = gradient_noise
        self.checkpointer = checkpoints.Checkpointer(**checkpointer)
        self.checkpoint_frequency_within_epoch = checkpoint_frequency_within_epoch

    def _log_train_batch(self, loss, stats, optimizer):
        logger.log_scalar('_loss', loss.item())
        for k, v in stats.items():
            is_scalar, scalar_val = utils.maybe_get_scalar(v)
            if is_scalar:
                logger.log_scalar('_' + k, scalar_val)
            else:
                utils.log(f"Could not log stat '{k}' to Tensorboard, since it is not a scalar or equivalent.",
                          level=logging.WARNING,
                          once=True)

        for i, param_group in enumerate(optimizer.param_groups):
            logger.log_scalar('_learning_rate{}'.format(i), param_group['lr'])

    def run(self, save_dir, model, train_dataset, eval_datasets=None,
            saved_state=None, debug_skip_training=False,
            probe_train_data=None):
        if saved_state:
            model.load_state_dict(saved_state['state_dict'])
            for k in saved_state:
                if k.startswith('avg_state_dict'):
                    print("Loading poyak's ", k)
                    setattr(model, k, saved_state[k])
        if eval_datasets is None:
            eval_datasets = {}
        if Globals.cuda:
            model.cuda()
            GPUs = [f"{i}) {torch.cuda.get_device_name(i)}"
                    for i in range(torch.cuda.device_count())]
            print(f"Trainer using GPUs: {','.join(GPUs)}.")
        else:
            print("Trainer not using GPU.")
        proto = getattr(torch.optim, self.optimizer_name)

        if self.codebook_lr is not None:
            optimizer = proto(
                [{'params': model.bottleneck.embedding.parameters(),
                  'lr': self.codebook_lr},
                 {'params': model.get_parameters_for_optimizer(with_codebook=False),
                  'lr': self.learning_rate}],
                lr=self.learning_rate, **self.optimizer_kwargs)
        else:
            optimizer = proto(
                model.get_parameters_for_optimizer(with_codebook=True),
                lr=self.learning_rate, **self.optimizer_kwargs)

        self.lr_scheduler_params['optimizer'] = optimizer
        lr_scheduler = utils.construct_from_kwargs(self.lr_scheduler_params)
        if saved_state:
            optimizer.load_state_dict(saved_state['optimizer'])
            lr_scheduler.load_state_dict(saved_state['lr_scheduler'])
        print(f"Optimizer: {optimizer}")

        if saved_state:
            self.current_iteration = saved_state['current_iteration']
            start_epoch = saved_state['epoch'] + 1
        else:
            self.current_iteration = 0
            start_epoch = 1

        self.checkpointer.set_save_dir(save_dir)

        if self.log_layers_stats:
            self.dbg = DebugStats.attach(model, logger)

        for epoch in range(start_epoch, self.num_epochs+1):
            Globals.epoch = epoch
            self.iterate_epoch(
                epoch, save_dir, model, train_dataset, eval_datasets,
                optimizer, lr_scheduler,
                debug_skip_training=debug_skip_training)

        if probe_train_data is not None:
            print(f"re-train all probes")
            probe_parameters = []
            for p in model.parameters():
                p.requires_grad = False
            for _, probe in model.probes.items():
                print(probe)
                for name,layer in probe.named_modules():
                    has_parameters = False
                    for name, param in layer.named_parameters():
                        if not "." in name:
                            has_parameters = True
                    if not has_parameters:
                        continue
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                    else:
                        print("WARNING: skip layer {0}".format(name))
                probe_parameters.extend(list(probe.parameters()))
                for name, param in probe.named_parameters():
                    print("re-train {0}".format(name))
                    param.requires_grad = True
            optimizer = proto(probe_parameters,
                              lr=self.learning_rate, **self.optimizer_kwargs)
            print("learning rate set to {0}".format(self.learning_rate))
            self.lr_scheduler_params['optimizer'] = optimizer
            lr_scheduler = utils.construct_from_kwargs(self.lr_scheduler_params)
            tmp = probe_train_data
            self.checkpointer.enabled = False

            logger.end_log()
            for epoch in range(1, 11):
                Globals.epoch = epoch
                #with torch.backends.cudnn.flags(enabled=False):
                self.iterate_epoch(
                    epoch, save_dir + "/probe_train/", model, probe_train_data, eval_datasets,
                    optimizer, lr_scheduler,
                    debug_skip_training=debug_skip_training,
                    only_train_probes=True)


    def iterate_epoch(self, epoch, save_dir, model, train_dataset,
                      eval_datasets, optimizer, lr_scheduler,
                      debug_skip_training=False, only_train_probes=False):
        data_len = len(train_dataset)
        if not only_train_probes:
            model.train()
        else:
            model.eval()
            for _, probe in model.probes.items():
                probe.train()
        self.setup_polyak_decay(model)
        progress = Progress()
        if not isinstance(lr_scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step()

        for batch_ind, batch in enumerate(train_dataset):
            self.current_iteration += 1
            Globals.current_iteration = self.current_iteration
            progress.announce_batch(
                self.current_iteration, epoch, batch_ind + 1, data_len)
            if (self.current_iteration % self.log_frequency) == 0:
                logger.make_step_log('{}/train/'.format(save_dir),
                                     self.current_iteration)
            else:
                logger.make_null_log()

            weight_noise_values = self.apply_temp_weight_noise(model)

            if Globals.cuda:
                batch = model.batch_to_device(batch, 'cuda')

            #with summary.Summary(model):
            if 1:
                loss, stats = model.minibatch_loss(batch)

            if self.kill_on_nan:
                self.check_loss(loss)

            optimizer.zero_grad()
            loss.backward()

            if self.log_layers_stats:
                self.dbg.show()
                self.dbg.reset()

            apply_opt_step = True
            if self.gradient_clipper:
                apply_opt_step = not self.gradient_clipper.clip(model)

            self.apply_gradient_noise(optimizer)
            self.undo_temp_weight_noise(model, weight_noise_values)
            if apply_opt_step:
                optimizer.step()
                self.calculate_polyak_decay(model)
            else:
                print("Trainer step skipped!")

            self._log_train_batch(loss, stats, optimizer)

            logger.end_log()

            progress.update(
                optimizer.param_groups[0]['lr'], loss, stats,
                flush=(self.current_iteration % self.output_frequency == 0))

            if (self.checkpoint_frequency_within_epoch and
                    batch_ind == self.checkpoint_frequency_within_epoch):
                print('Saving checkpoint...')
                self.checkpointer.checkpoint(self.current_iteration, epoch,
                                             model, optimizer, lr_scheduler)

            if debug_skip_training:
                break

        print('Epoch {} ended.'.format(epoch))

        if self.checkpointer.enabled:
            print("  + saving last epoch")
            self.checkpointer.checkpoint(self.current_iteration, epoch,
                                         model, optimizer, lr_scheduler)

        eval_results = self.evaluate(model, eval_datasets, save_dir)
        if isinstance(lr_scheduler,
                      torch.optim.lr_scheduler.ReduceLROnPlateau):
            if 'loss' in eval_results.get('dev', {}):
                # Don't decay the LR during noise rampup
                wn_allows_lr_red = (
                    self.weight_noise == 0 or
                    (self.weight_noise_start_iteration <
                     self.current_iteration)
                    or not self.weight_noise_prevent_lr_step)
                if wn_allows_lr_red:
                    lr_scheduler.step(metrics=eval_results['dev']['loss'],
                                      epoch=epoch)
            else:
                print("Warning: evaluation didn't yield a dev loss, can't"
                      "step the optimizer.")
        for subset in eval_results:
            for key, outcome in eval_results[subset].items():
                self.checkpointer.try_checkpoint_best(
                    (subset + '_' + key).replace('_', '-').replace("[%]","").replace("/",""),
                    outcome, # Typically, float value
                    self.current_iteration,
                    epoch, model, optimizer, lr_scheduler)

        elapsed = datetime.now() - progress.start_time
        print("Current epoch took {}".format(elapsed))

    # dataset == actually a DatasetLoader, not a DataSet
    def evaluate(self, model, eval_datasets, save_dir, eval_subset=""):
        def iter_batches(dataset, log_dir, iteration):
            logger.make_step_log(log_dir, iteration)
            tic = NEGINF
            for batch_i, batch in enumerate(dataset):
                if time.time() - tic > 30:
                    num_examples = len(next(iter(batch.values())))
                    print('Processing batch {}/{} ({} elements)'.format(
                        batch_i, len(dataset), num_examples))
                    tic = time.time()
                if Globals.cuda:
                    batch = model.batch_to_device(batch, 'cuda')
                yield batch
                # This logs only the first batch of the test data
                if logger.is_currently_logging():
                    logger.end_log()
                    logger.make_null_log()
            logger.end_log()
            logger.make_step_log(log_dir, iteration)

        def eval_single_subset(log_name, dataset):
            print('Evaluating model on {}'.format(log_name))
            with torch.no_grad():
                results = model.evaluate(iter_batches(
                    dataset, log_dir='{}/{}/'.format(save_dir, log_name),
                    iteration=self.current_iteration))
            print('Results on {}: {}'.format(log_name, results))
            for k, v in results.items():
                logger.log_scalar('_' + k, v)
            logger.end_log()
            return results

        model.eval()
        results = {}
        for k in eval_datasets:

            # If you provided a specific subset name to evaluate then only do that one.
            if eval_subset != "" and k != eval_subset:
                print("evaluate() Skipping eval dataset named: " + str(k) )
                continue

            results[k] = eval_single_subset(k, eval_datasets[k])

            if hasattr(model, 'avg_state_dict'):
                old_state = deepcopy(model.state_dict())
                model.load_state_dict(model.avg_state_dict)
                name = '{}_polyak'.format(k)
                results[name] = eval_single_subset(
                    name, eval_datasets[k])
                model.load_state_dict(old_state)

            if isinstance(self.polyak_decay, list):
                old_state = deepcopy(model.state_dict())
                for decay in self.polyak_decay:
                    name = '{}_polyak_{}'.format(k, decay)
                    polyak_dict_name = 'avg_state_dict_%f' % (decay)
                    try:
                        polyak_dict = getattr(model, polyak_dict_name)
                        model.load_state_dict(polyak_dict)
                        results[name] = eval_single_subset(
                            name, eval_datasets[k])
                    except:
                        print ("evaluate() No avg_state_dict_ polyak dict found, skip." )
                model.load_state_dict(old_state)

        return results

    def setup_polyak_decay(self, model):
        if not self.polyak_decay:
            return
        if isinstance(self.polyak_decay, list):
            if hasattr(model, 'avg_state_dict'):
                st_dict = model.avg_state_dict
            else:
                st_dict = model.state_dict()
            good_polyak_names = []
            for decay in self.polyak_decay:
                polyak_dict_name = 'avg_state_dict_%f' % (decay)
                good_polyak_names.append(polyak_dict_name)
                if not hasattr(model, polyak_dict_name):
                    setattr(model, polyak_dict_name, deepcopy(st_dict))
            for polyak_dict_name in list(model.__dict__.keys()):
                if (polyak_dict_name.startswith('avg_state_dict') and
                        polyak_dict_name not in good_polyak_names):
                    print("Polyak deleting ", polyak_dict_name)
                    delattr(model, polyak_dict_name)
        elif self.polyak_decay > 0 and \
                not hasattr(model, 'avg_state_dict'):
            model.avg_state_dict = deepcopy(model.state_dict())

    def calculate_polyak_decay(self, model):
        if not self.polyak_decay:
            return
        st = model.state_dict()
        if isinstance(self.polyak_decay, list):
            for decay in self.polyak_decay:
                polyak_dict_name = 'avg_state_dict_%f' % (decay)
                polyak_dict = getattr(model, polyak_dict_name)
                for k in polyak_dict:
                    polyak_dict[k] = (
                        decay*polyak_dict[k] +
                        (1 - decay)*st[k])
        elif self.polyak_decay > 0:
            for k in model.avg_state_dict:
                model.avg_state_dict[k] = (
                    self.polyak_decay*model.avg_state_dict[k] +
                    (1 - self.polyak_decay)*st[k])

    def check_loss(self, final_loss):
        if torch.isnan(final_loss).item():
            print('Loss is nan. Killing...')
            exit(1)
        if final_loss.item() == INF or final_loss.item() == NEGINF:
            print('Loss is inf. Killing...')
            exit(1)

    def apply_gradient_noise(self, optimizer):
        if not self.gradient_noise:
            return
        var = self.gradient_noise / (1 + self.current_iteration)**0.55
        var = var**2
        for param_group in optimizer.param_groups:
            for parameter in param_group['params']:
                parameter.grad += (
                    torch.randn_like(parameter.grad) * var)

    def apply_temp_weight_noise(self, model):
        apply_weight_noise = (
            self.weight_noise > 0 and
            (self.current_iteration > self.weight_noise_start_iteration or
             self.weight_noise_linear_increase))

        noise_values = {}
        if apply_weight_noise:
            for name, weight in model.named_parameters():
                if 'weight' in name and 'batch_norm' not in name:
                    incr = 1.0
                    if self.weight_noise_linear_increase:
                        start = max(1, self.weight_noise_start_iteration)
                        incr = min(incr, self.current_iteration / start)
                    rand = (incr * self.weight_noise
                            * torch.randn_like(weight))
                    noise_values[name] = rand
                    weight.data.add_(rand)
        return noise_values

    def undo_temp_weight_noise(self, model, noise_values):
        if noise_values:
            for name, weight in model.named_parameters():
                if 'weight' in name and 'batch_norm' not in name:
                    weight.data.add_(-noise_values[name])
