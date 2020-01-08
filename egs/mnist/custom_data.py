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

import torch
from torch.utils.data import Dataset
import pickle


class MnistDataSet(Dataset):
    # This is used to train SOM-VAE from [1]
    # [1] Fortuin, Vincent et al.
    # “SOM-VAE: Interpretable Discrete Representation Learning on Time Series.” ICLR 2019 (2019).
    # To reproduce the results we would like to use the exact same dataset used in the paper
    def __init__(self, data_path='data/mnist.somvae.pkl', train=False):
        self.train = train
        mnist = pickle.load(open(data_path, "rb"))
        data_train = mnist["data_train"]
        labels_train = mnist["labels_train"]
        # dataset
        self.xte = data_train[45000:]
        self.yte = labels_train[45000:]
        self.xtr = data_train[:45000]
        self.ytr = labels_train[:45000]

    def __len__(self):
        if self.train:
            return self.xtr.shape[0]
        else:
            return self.xte.shape[0]

    def __getitem__(self, item):
        if self.train:
            x = torch.from_numpy(self.xtr[item]).float()
            y = torch.argmax(torch.from_numpy(self.ytr[item]).long(), dim=-1)
        else:
            x = torch.from_numpy(self.xte[item]).float()
            y = torch.argmax(torch.from_numpy(self.yte[item]).long(), dim=-1)  # don't need one hot
        return {"features": x, "targets": y}
