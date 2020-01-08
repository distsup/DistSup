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

import logging
from functools import reduce

import torch
import torch.nn

from distsup import utils
from distsup.models import base
from distsup.modules.aux_modules import attach_auxiliary

mod_logger = logger = logging.getLogger(__name__)


class ProbedNet(base.Model):
    """
    A model that allows to easily evaluate the capacity of layers' outputs to predict targets in the batch.

    This works by attaching auxiliary heads (probes) at several points in the net that
    are trained to predict a desired target from the batch and the accuracies/losses of these will be reported.

    Warning: the heads are trained at the same time as the network (at each forward) so the dynamics of training
    the main net can affect the prediction ability of the probes, making the results less comparable across models.

    You may use this class by doing as follows:

    class MyModel(ProbedNet):
        def __init__(self, **kwargs):
            super().__init__(self, **kwargs)

            # Here model construction

            self.add_probes()

        def minibatch_loss(self, batch):
            # Here model loss

            raw_features = batch['features']    # ensure B x T (or W) x H x C
            probes_losses, probes_details = self.probes_loss(raw_features, batch)
            details.update(probes_details)

            return loss + probes_losses, details

    """
    def __init__(self, probes=None, aux_heads=None, **kwargs):
        super(ProbedNet, self).__init__(**kwargs)

        # We intialize to None in order to check if the probes have been initialized or not
        self.probes = None
        self.probes_dict = probes or {}
        for probe in self.probes_dict.values():
            if probe.get('bp_to_main', False):
                raise Exception("Probes should not influence the model")
        if aux_heads:
            for k, v in aux_heads.items():
                if k in self.probes_dict:
                    raise Exception("Aux head duplicates probe name")
                self.probes_dict[k] = v
        self.probes_test_forward = True

    def test_batch_forward(self):
        if self.dataloader is None:
            raise ValueError('ProbedNet requires access to the dataloader used for training in order '
                             'to do a test run and gather layer output dimensions required to attach heads.'
                             'Pass the keyword argument dataloader=my_dataloader to the model constructor.')

        batch = next(iter(self.dataloader))

        self.probes_test_forward = True
        self.eval()

        with torch.no_grad():
            result = self.minibatch_loss(batch)

        self.probes_test_forward = False
        self.train()

        return batch, result

    def add_probes(self):
        """Method to attach the probes to the models.

        :param probes: list of dict with the probes to attach. Keys are the probe names, values are dicts containing:
         - layer: str the name of the member variable the output on which to run the predictor,
         - target: str the field of the batch where to find the target that needs to be predicted
         - predictor: dict with the the predictor to be added
         - **kwargs: other kwargs of the probe class
        :return:
        """

        probes_dict = self.probes_dict
        self.probes_dict = {}
        self.probes = torch.nn.ModuleDict()

        def _register_output_shape(module, _, mod_output, name):
            if isinstance(mod_output, (list, tuple)):
                mod_output = mod_output[0]

            if isinstance(mod_output, torch.Tensor):
                module.output_shape = mod_output.shape

            else:
                mod_logger.warning(f'Could not gather shape of module {name} / output #0')

        probe_hooks = []
        for name, m in self.named_modules():
            probe_hooks.append(m.register_forward_hook(lambda mod, in_mod, out_mod, name=name:
                                                       _register_output_shape(mod, in_mod, out_mod, name)))

        test_batch, _ = self.test_batch_forward()

        for hook in probe_hooks:
            hook.remove()

        # Attach each probe to its place and build the predictors
        for probe_name, probe_dict in probes_dict.items():
            required_keys = {'layer', 'target', 'predictor'}
            if required_keys - probe_dict.keys():
                mod_logger.error(f"Expecting the following keys in the '{probe_name}' probe"
                                 f" dictionary: {required_keys}. Only found: {probe_dict.keys()}."
                                 f"Ignoring probe {probe_name}.")
            all_known_keys = required_keys | {'bp_to_main', 'learning_rate',
                                              'which_out', 'requires'}
            if set(probe_dict.keys()) - all_known_keys:
                raise Exception(f'Probe {probe_name} has unsupported arguments: '
                                f'{set(probe_dict.keys()) - all_known_keys}.')

            mod_logger.info(f'Adding probe {probe_name} to {probe_dict["layer"]}.')

            layer = self.get_named_module(probe_dict['layer'], probe_name)

            if probe_dict['target'] not in test_batch:
                mod_logger.error(f'Could not found target named {probe_dict["target"]} in test batch. '
                                 f'Available keys are: {test_batch.keys()}.'
                                 f'Ignoring probe {probe_name}')

            if len(layer.output_shape) != 4:
                mod_logger.error(f'Expecting data layout of the layer to be B x W x H x C.'
                                 f'The two last dimensions will be flatten when fed to the probe.'
                                 f'Currently obtained {layer.output_shape}')

            input_dim = layer.output_shape[-2] * layer.output_shape[-1]

            additional_predictor_parameters = {'input_dim': input_dim}

            if 'requires' in probe_dict:
                try:
                    # Recursively retrieve, e.g. self.bottleneck.num_tokens
                    val = reduce(lambda obj,attr: getattr(obj, attr),
                                 probe_dict['requires'].split('.'), self)
                    name = probe_dict['requires'].split('.')[-1]
                    additional_predictor_parameters.update({name: val})
                except AttributeError:
                    logger.warning(f"{probe_name} disabled; "
                                   f"{probe_dict['requires']} not available")
                    continue

            dataloader = self.dataloader
            if (hasattr(dataloader, 'metadata') and
                    (probe_dict['target'] in dataloader.metadata) and
                    (dataloader.metadata[probe_dict['target']]['type'] == 'categorical')):
                target_size = dataloader.metadata[probe_dict['target']]['num_categories']
                additional_predictor_parameters['output_dim'] = target_size

            predictor_dict = probe_dict['predictor']
            predictor = utils.construct_from_kwargs(predictor_dict,
                                                    additional_parameters=additional_predictor_parameters)
            probe = attach_auxiliary(layer,
                                     predictor,
                                     bp_to_main=probe_dict.get('bp_to_main', False),
                                     which_out=probe_dict.get('which_out', 0))

            self.probes_dict[probe_name] = probe_dict
            self.probes[probe_name] = probe

    def get_named_module(self, module_name, probe_name):
        module_names = dict(self.named_modules())

        if module_name not in module_names:
            mod_logger.error(f'Could not find layer named {module_name}. '
                             f'Available layers: {module_names.keys()}. '
                             f'Ignoring probe {probe_name}.')

            return None

        return module_names[module_name]

    def probes_loss(self, batch):
        """ Compute the probes losses.
        :param batch: batch where to find the target of the probe predictor. Should have same T as raw_features.
        :return: sum reduced loss of the probes and details dict containing any stats
        """

        if self.probes_test_forward:
            return torch.tensor(0), torch.tensor(0), {}

        if self.probes_dict and self.probes is None:
            raise ValueError('The probes have not been added. Be sure to add a call to '
                             'self.add_probes(probes_dictionary) at the end of your model\'s constructor.')

        detached_losses = []
        backpropagating_losses = []
        details = {}

        for probe_name, probe in self.probes.items():
            target_name = self.probes_dict[probe_name]['target']
            probe_loss, probe_details = probe.loss(
                batch['features'],
                batch[target_name],
                features_len=batch.get('features_len'),
                targets_len=batch.get(f'{target_name}_len'))

            probe_loss *= self.probes_dict[probe_name].get('learning_rate', 1.0)
            if self.probes_dict[probe_name].get('bp_to_main'):
                backpropagating_losses.append(probe_loss)
            else:
                detached_losses.append(probe_loss)
            probe_details['loss'] = probe_loss

            details.update({f'{probe_name}_{k}': v for k, v in probe_details.items()})

        return sum(detached_losses), sum(backpropagating_losses), details


def main():
    pass


if __name__ == '__main__':
    main()
