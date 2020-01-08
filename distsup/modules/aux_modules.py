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

import logging

mod_logger = logger = logging.getLogger(__name__)


def _pick_nth(tensor_or_sequence, which=0):
    if isinstance(tensor_or_sequence, (list, tuple)):
        tensor_or_sequence = tensor_or_sequence[which]
    else:
        if which > 0:
            raise ValueError("Requested output not present")
    return tensor_or_sequence


def _detach(tensor_or_iterable):
    if isinstance(tensor_or_iterable, (list, tuple)):
        return [_detach(elem) for elem in tensor_or_iterable]
    elif isinstance(tensor_or_iterable, dict):
        return {k: _detach(v) for k, v in tensor_or_iterable.items()}
    else:
        return tensor_or_iterable.detach()


def store_input_for_aux(f, which=0):
    def fun(mod, main_in, main_out):
        f.input = _pick_nth(main_out, which)
    return fun


def store_detached_input_for_aux(f, which=0):
    def fun(mod, main_in, main_out):
        # Note to future self: if we ever decide to pass all that we captured
        # to the aux module, then make sure to recourse all tuples, lists,
        # dicts and other iterables to detach all tensors!!
        f.input = _detach(_pick_nth(main_out, which))
    return fun


def attach_auxiliary(main, aux, bp_to_main=False, which_out=0):
    """
    An auxiliary module can be attached to a main module. When that main module
    performs its forward pass, the auxiliary module is called on its output.
    The output can be gathered later.

    Example:
        m = Model()
        clf = attach_auxiliary(m.layer_3, SomeClassifier())


        out = m(...)
        loss = foo(out) + clf.out
        ...

    Args:
        main: the module aux will be attached to
        aux: the attached module to main
        bp_to_main (bool): whether the loss on aux should backprop to the
        main module
    """
    if bp_to_main:
        mod_logger.info("Adding aux module that backprops.")
        main.register_forward_hook(store_input_for_aux(aux, which=which_out))
    else:
        mod_logger.info("Adding aux module that does not backprop.")
        main.register_forward_hook(
            store_detached_input_for_aux(aux, which=which_out))
    return aux
