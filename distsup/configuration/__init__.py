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

import torch

from distsup import utils
from distsup.configuration import config_utils


def get_val(dictionary, key, dict_name):
    if key not in dictionary:
        raise KeyError('%s has no %s key specified' % (dict_name, key))

    return dictionary[key]


class ConfigInstantiator(object):
    def __init__(self, objects_config, default_class_dict={},
                 default_modules_dict={}, name='', **kwargs):
        super(ConfigInstantiator, self).__init__(**kwargs)
        self.objects_config = objects_config
        self.default_class_dict = default_class_dict
        self.default_modules_dict = default_modules_dict
        self.cache = {}
        self.name = name

    def keys(self):
        return self.objects_config.keys()

    def _getitem(self, key, additional_parameters=None):
        if key not in self.cache:
            # make a copy since we may change the dict in the end
            opts = dict(get_val(self.objects_config, key, self.name))
            if 'class_name' not in opts:
                opts['class_name'] = self.default_class_dict[key]
            self.cache[key] = utils.construct_from_kwargs(
                    opts, self.default_modules_dict.get(key),
                    additional_parameters)
        return self.cache[key]

    def __getitem__(self, key):
        return self._getitem(key)


class DatasetConfigInstantiator(ConfigInstantiator):
    def _getitem(self, key, additional_parameters=None):
        if key not in self.cache:
            # make a copy since we may change the dict in the end
            opts = dict(get_val(self.objects_config, key, self.name))
            if 'class_name' not in opts:
                opts['class_name'] = self.default_class_dict[key]
            self.cache[key] = utils.construct_from_kwargs(
                    opts, self.default_modules_dict.get(key),
                    additional_parameters)
        return self.cache[key]


class _ConstantDict(object):
    def __init__(self, v, **kwargs):
        super(_ConstantDict, self).__init__(**kwargs)
        self.v = v

    def __getitem__(self, k):
        return self.v

    def get(self, k, v=None):
        return self.v


class Configuration(ConfigInstantiator):
    """
    Class responsible for instantiating object that are defined in config file.

    The class tries to be smart about the following modules:
    - Trainer will by default instantiate an 'distsup.trainer.Trainer'
    - all items on the Data key will instantiate a 'distsup.data.Data'
    - It will configure the Model key according to Dataset specification

    Args:
        config_path (str): Path pointing to the config file.
        modify_dict (dict): Optional dictionary representing config
            modifications.
        store_path (str): Optional path to store linked config.
    """

    default_class_dict = {
        'Trainer': 'Trainer',
    }
    default_modules_dict = {
        'Trainer': 'distsup.trainer',
        'Datasets': 'distsup.data',
        'Model': 'models',
    }

    def __init__(self, config_path, modify_dict={}, store_path=None, **kwargs):
        config = config_utils.ConfigParser(config_path).get_config(modify_dict)
        if store_path is not None:
            config_utils.ConfigLinker(config).save_linked_config(store_path)
        super(Configuration, self).__init__(
            objects_config=config,
            default_class_dict=Configuration.default_class_dict,
            default_modules_dict=Configuration.default_modules_dict,
            name=config_path,
            **kwargs)
        if 'Datasets' in self.objects_config:
            self.cache['Datasets'] = DatasetConfigInstantiator(
                self.objects_config['Datasets'],
                default_modules_dict=_ConstantDict(
                        Configuration.default_modules_dict['Datasets']),
                name='Config.Datasets')

    def __getitem__(self, key):
        if key == 'Model':
            model_param = {'dataloader': self['Datasets']['train']}
            return self._getitem('Model', additional_parameters=model_param)
        else:
            return super(Configuration, self).__getitem__(key)


class Globals(object):
    """Global configuration objects."""
    cuda = torch.cuda.is_available()
    cluster = ''
    exp_tag = None
    save_dir = None
    exp_uuid = None
    exp_config_fpath = None

    # Track training progress. The trainer/loader will fill in proper values.
    epoch = -1
    current_iteration = -1
