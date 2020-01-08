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
from . import TensorLogger


class DefaultTensorLogger(object):
    log_instance = None

    def __init__(self):
        if DefaultTensorLogger.log_instance is None:
            DefaultTensorLogger.log_instance = TensorLogger()

    def __getattr__(self, name):
        return getattr(self.log_instance, name)
