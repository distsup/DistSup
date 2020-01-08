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

# model summary as in https://github.com/sksq96/pytorch-summary

import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np


class Summary():
    def __init__(self, model):
        if hasattr(model, '_summary_printed'):
            self.enabled = False
            return
        else:
            self.enabled = True
            self.model = model
            model._summary_printed = True

    def __enter__(self):
        if not self.enabled:
            return

        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                if isinstance(input[0], (list, tuple)):
                    summary[m_key]["input_shape"] = [
                        list(o.size()) if hasattr(o, 'size') else "??" for o in input[0]
                    ]
                else:
                    summary[m_key]["input_shape"] = list(input[0].size())
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        list(o.size()) if hasattr(o, 'size') else "??" for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == self.model)
            ):
                hooks.append(module.register_forward_hook(hook))

        # create properties
        summary = OrderedDict()
        hooks = []

        # register hook
        self.model.apply(register_hook)

        self.hooks = hooks
        self.summary = summary

    def __exit__(self, type, value, traceback):
        if not self.enabled:
            return

        hooks = self.hooks
        summary = self.summary
        for h in hooks:
            h.remove()

        print("----------------------------------------------------------------")
        line_new = "{:>20} {:>25} -> {:>25} {:>15}".format("Layer (type)", "Input Shape", "Output Shape", "Param #")
        print(line_new)
        print("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20} {:>25} -> {:>25} {:>15}".format(
                layer,
                str(summary[layer]["input_shape"]),
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            #total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"]:
                    trainable_params += summary[layer]["nb_params"]
            print(line_new)

        # assume 4 bytes/number (float on cuda).
        # total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
        # total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size #+ total_output_size  # + total_input_size

        print("================================================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(
            total_params - trainable_params))
        print("----------------------------------------------------------------")
        # print("Input size (MB): %0.2f" % total_input_size)
        #print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("----------------------------------------------------------------")
