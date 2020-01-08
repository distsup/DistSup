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

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

class GruVariableLength(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(GruVariableLength, self).__init__(**kwargs)
        self.gru_hidden_dim = output_dim
        self.encoder_output_dim = input_dim
        self.gru = nn.GRU(self.encoder_output_dim, self.gru_hidden_dim, 1)

    def forward(self, features, features_len=None):
        if features_len is None:
            features_len = torch.ones(
                features.size(0), dtype=torch.int32, device=features.device
            ) * features.size(1)
        # pack_padded need TxBx*
        f = features.contiguous().view(features.size(0), features.size(1), -1)
        f = f.permute(1, 0, 2)
        packedSeq = torch.nn.utils.rnn.pack_padded_sequence(
            f, features_len, enforce_sorted=False
        )
        (out, hn) = self.gru(packedSeq)
        (ogru, ogru_l) = torch.nn.utils.rnn.pad_packed_sequence(out)
        ogru = ogru.permute(1, 0, 2).unsqueeze(2)
        return ogru, ogru_l


class CPCModule(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        input_dim_force=None,
        gru_hidden_dim=64,
        reduction="mean",
        k=6,
        N=5,
        compute_kcer=False,
        loss_details=False,
        permute=None,
        bias=True,
        k_step=1,
        n_around=100,
    ):
        super(CPCModule, self).__init__()

        self.reduction = reduction
        self.n_around = n_around
        self.gru_hidden_dim = gru_hidden_dim
        if input_dim_force is not None:
            input_dim = input_dim_force

        self.gru_input_dim = input_dim

        # k is the number of frames we want to predict
        self.k = k
        self.k_step = k_step
        # N is the number of negative samples taken for each positive one
        self.N = N
        # list of permute
        self.permute = permute

        self.gru = GruVariableLength(self.gru_input_dim, self.gru_hidden_dim)
        self.cpc_compute_kcer = compute_kcer
        self.loss_details = loss_details

        self.W = nn.ModuleList()

        for i in range(k):
            # W0 is actually the log bilinear model for the prediction of t+1
            self.W.append(
                nn.Bilinear(
                    self.gru_input_dim, self.gru_hidden_dim, 1, bias=False
                )
            )

    def __repr__(self):
        fmt_str = "%i pred, %i noise" % (self.k, self.N)
        fmt_str += " (GRU input_dim=%i, hidden_dim=%i, compute_kcer=%s)" % (
            self.gru_input_dim,
            self.gru_hidden_dim,
            self.cpc_compute_kcer,
        )
        return fmt_str

    def _isInf(self, value):
        if (
            value.item() == float("inf")
            or value.item() == float("-inf")
            or torch.isnan(value)
        ):
            return True
        return False

    def forward(self, features, features_len=None):
        (ct, ogru_length) = self.gru(features, features_len)
        return (ct, features, ogru_length)

    def cpc_loss(self, gru_input_feats, gru_output_feats, feats_len):
        zt_feats = gru_input_feats.contiguous().view(
            gru_input_feats.size(0), gru_input_feats.size(1), -1
        )
        ct = gru_output_feats.contiguous().view(
            gru_output_feats.size(0), gru_output_feats.size(1), -1
        )
        zt_length = feats_len

        tot_loss = 0
        nb_examples = 0
        lossK = {}  # key=k, value=(tot_loss_k, nb_example_k)
        nbErrK = {}  # key=k, value=(tot_err_k, nb_example_k)

        # change from BxTx(FxC) to TxBxF
        zt_feats = zt_feats.permute(1, 0, 2)
        ct = ct.permute(1, 0, 2)

        for b in range(zt_length.size(0)):
            seq_len = zt_length[b].item()

            # compute indices
            matK = np.arange(self.k + 1)[
                :, np.newaxis
            ] + np.arange(0, seq_len)
            # example:
            # ct_i    (0 1 2 3 4 5 6)
            # zt_i k0 (1 2 3 4 5 6 7)
            # zt_i k1 (2 3 4 5 6 7 8)
            # ...

            noise = min(self.N, seq_len - 1)
            noiseC_ind = np.arange(seq_len * noise) // noise
            # if noise = 3, produce (0 0 0 1 1 1 2 2 2 ...)

            for k in range(self.k_step, self.k, self.k_step):
                z_ind = matK[k][matK[k] < seq_len]
                # for example if k=1, z_ind = (2 3 4 5 ... seq_len-1)
                in_feats_z = zt_feats[z_ind, b]
                # then select the zt_feats corresponding to thoses indices
                in_feats_c = ct[matK[0][: in_feats_z.size(0)], b]
                # and the ct_feats correesponding to the first line of matK
                # limited to the number of values in z_feats
                # then we wants to learn W as f(x_1, c_1) = exp(z_1.T W_1 c_1)

                noiseInd = np.zeros((in_feats_z.size(0), noise))

                for i, z in enumerate(z_ind):
                    rand = np.random.permutation(seq_len)
                    orig = rand[rand != z]
                    rand = orig[orig < (z+self.n_around)]
                    rand = rand[rand > (z-self.n_around)]
                    if(rand.shape[0] >= noise):
                        n_indices = rand[:noise]
                    else:
                        n_around = noise+1
                        rand = orig[orig < (z+n_around)]
                        rand = rand[rand > (z-n_around)]
                        n_indices = rand[:noise]
                    # taking random indices (different to the one of the posit.
                    # z_feat, limited to the number of noise that we want
                    noiseInd[i] = n_indices
                    # for each value of z we have noise random indices
                    # in noiseInd matrix

                # noiseC_ind contains  (0 0 0 ...)
                # noise_Ind  contains  (rand(seq_len)!=z rand(seq_len)!=z ...)
                noise_feats_z = zt_feats[
                    noiseInd.reshape(in_feats_z.size(0) * noise), b
                ]
                noise_feats_c = ct[noiseC_ind[: in_feats_z.size(0) * noise], b]

                fxt_all = self.W[k](
                    torch.cat((in_feats_z, noise_feats_z), 0),
                    torch.cat((in_feats_c, noise_feats_c), 0),
                )

                f_x_t_k = fxt_all[: in_feats_z.size(0)]

                # loss = -(torch.log(f_x_t_k.exp() / fxt_all.exp().sum()).sum())
                loss = -(f_x_t_k - torch.logsumexp(fxt_all, dim=0)).sum()

                slen = in_feats_z.size(0)

                if self.cpc_compute_kcer:
                    # classify each elem of the sequence to compute the cer
                    nbErr = 0
                    for pred in range(slen):
                        in_feats = (in_feats_c[pred], in_feats_z[pred])
                        offset = pred * noise
                        noise_f = (
                            noise_feats_c[offset : offset + noise],
                            noise_feats_z[offset : offset + noise],
                        )
                        f_x_t_n = F.bilinear(
                            torch.cat(
                                (in_feats[1].unsqueeze(0), noise_f[1]), 0
                            ),
                            torch.cat(
                                (in_feats[0].unsqueeze(0), noise_f[0]), 0
                            ),
                            self.W[k].weight,
                            self.W[k].bias,
                        )
                        probs = f_x_t_n.exp() / f_x_t_n.exp().sum()
                        nbErr += probs.argmax() != 0

                tot_loss += loss
                nb_examples += f_x_t_k.size(0)
                if k not in lossK:
                    lossK[k] = (loss, f_x_t_k.size(0))
                    if self.cpc_compute_kcer:
                        nbErrK[k] = (nbErr, f_x_t_k.size(0))
                else:
                    (tot_loss_k, nb_examples_k) = lossK[k]
                    nb_examples_k += f_x_t_k.size(0)
                    if self.cpc_compute_kcer:
                        (tot_errors, nbEx) = nbErrK[k]
                        tot_errors += nbErr
                        nbErrK[k] = (tot_errors, nb_examples_k)
                    tot_loss_k += loss
                    lossK[k] = (tot_loss_k, nb_examples_k)



        if self.reduction == "sum":
            tot_loss = 0
            for k in lossK.keys():
                (tot_loss_k, nb_examples_k) = lossK[k]
                tot_loss += tot_loss_k/nb_examples_k
        else:
            tot_loss = 0
            nb_examples = 0
            for k in lossK.keys():
                (tot_loss_k, nb_examples_k) = lossK[k]
                tot_loss += tot_loss_k
                nb_examples += nb_examples_k
            tot_loss /= nb_examples

        details = {}
        details["loss"] = tot_loss
        for k in lossK.keys():
            (tot_loss_k, nb_examples_k) = lossK[k]
            if self.cpc_compute_kcer:
                (nbErr, nbEx) = nbErrK[k]
                details["cer_k" + str(k + 1)] = (
                    torch.tensor(nbErr / nbEx) * 100
                )
            if self.loss_details:
                details["loss_k" + str(k + 1)] = tot_loss_k / nb_examples_k

        if tot_loss.item() == float("inf") or tot_loss.item() == float("-inf"):
            print("Inf loss !!")

        return tot_loss, details

    # TODO: possibly make it standalone function, that is called on from the CPC probe

    def loss(self, features, targets, features_len=None, targets_len=None):
        zt_feats = self.out
        if features_len is not None:
            rate_factor = features.size(1) // zt_feats.size(1)
            zt_length = (features_len + rate_factor - 1) // rate_factor
        else:
            zt_length = None

        ct = self(zt_feats, zt_length)

        return self.cpc_loss(zt_feats, ct, zt_length)