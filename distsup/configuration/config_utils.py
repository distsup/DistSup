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

import copy
import logging
import os
import sys

import yaml

from distsup import utils


class ConfigYamlLoader(yaml.FullLoader):
    def __init__(self, stream):
        self._stream_name = stream.name
        self._root = os.path.split(stream.name)[0]

        super(ConfigYamlLoader, self).__init__(stream)

    def include(self, node):
        path = self.construct_scalar(node)

        filename = os.path.expanduser(os.path.expandvars(path))
        if not os.path.isfile(filename):
            filename = os.path.normpath(os.path.join(self._root, filename))

        if not os.path.isfile(filename):
            logging.error(f'Could not find included yaml {path} in config file {self._stream_name}. '
                          f'It searched both in the current working directory and in the directory of the yaml file.')
            sys.exit(1)

        with open(filename, 'r') as f:
            return yaml.load(f, ConfigYamlLoader)


ConfigYamlLoader.add_constructor('!include', ConfigYamlLoader.include)


class ConfigLinker:
    def __init__(self, config_dict):
        self.config_dict = config_dict

    def save_linked_config(self, output_file):
        dir_name = os.path.dirname(output_file)
        if len(dir_name) > 0:
            utils.ensure_dir(dir_name)
        with open(output_file, 'w') as file_stream:
            yaml.dump(self.config_dict, file_stream, default_flow_style=False)


class ConfigParser:
    PARENT_NODE = 'parent'

    def __init__(self, root_config_file):
        self.root_config_file = root_config_file
        self.root_config_dict = None

    def apply_changes_in_config(self, config_dict, changes_dict):
        for key, value in changes_dict.items():
            if key in config_dict and isinstance(value, dict) and isinstance(config_dict[key], dict):
                self.apply_changes_in_config(config_dict[key], value)
            else:
                config_dict[key] = value

    def read_config(self, config_file):
        with open(config_file) as config_stream:
            config_dict = yaml.load(config_stream, Loader=ConfigYamlLoader)
            if self.PARENT_NODE in config_dict:
                raise Exception("Parent linking is not supported any more.")
                parent_file = os.path.expandvars(config_dict[self.PARENT_NODE])
                changes_dict = config_dict
                config_dict = self.read_config(parent_file)
                self.apply_changes_in_config(config_dict, changes_dict)
            return config_dict

    def init_config_dict(self):
        if self.root_config_dict is None:
            self.root_config_dict = self.read_config(self.root_config_file)

    def modify_config_node(self, config_dict, path, value):
        path_parts = path.split('.')
        path_node = config_dict
        for path_part in path_parts[:-1]:
            path_node = path_node[path_part]
        path_node[path_parts[-1]] = yaml.load(value, Loader=yaml.FullLoader)

    def get_config(self, modify_dict={}):
        self.init_config_dict()
        config_dict = copy.deepcopy(self.root_config_dict)
        for path, value in modify_dict.items():
            self.modify_config_node(config_dict, path, value)
        return config_dict
