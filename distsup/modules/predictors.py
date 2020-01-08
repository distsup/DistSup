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
import torch.nn as nn
import torch.nn.functional as F

from distsup import utils
from distsup.logger import DefaultTensorLogger
from distsup.modules.encoders import makeRnn, RNNStack

logger = DefaultTensorLogger()


class GlobalPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, aggreg=3, time_reduce="avg"):
        super(GlobalPredictor, self).__init__()
        self.pred = nn.Conv1d(input_dim, output_dim, kernel_size=aggreg)
        self.time_reduce = time_reduce

    def forward(self, vq_output, features_len, targets_len=None):
        vq_output = (vq_output
                     .contiguous()
                     .view(vq_output.size(0), vq_output.size(1), -1)
                     .permute(0, 2, 1))
        vq_output = F.pad(vq_output, (0, self.pred.kernel_size[0] - 1, 0, 0))
        out_conv = self.pred(vq_output)
        mask = utils.get_mask1d(features_len, mask_length=out_conv.size(2))
        mask.unsqueeze_(1)

        avg_mask = mask / mask.sum()

        if self.time_reduce == "avg":
            return (out_conv * avg_mask).sum(dim=2)
        elif self.time_reduce == "max":
            return torch.where(out_conv == 1, mask, out_conv.min()).max(dim=2)[0]
        else:
            raise NotImplementedError(
                "GlobalPredictor: not a valid reduction:" + self.time_reduce
            )

    def loss(self, features, targets, features_len=None, targets_len=None):
        out = self(self.input, features_len)
        loss = F.cross_entropy(out, targets)
        predicted = out.argmax(dim=1, keepdim=True)
        acc = torch.mean((targets == predicted).float())
        return loss, {'acc': acc}


class TripleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, aggreg=3, only=None):
        super(TripleClassifier, self).__init__()
        self.only = only

        class LstmClf_(nn.Module):
            def __init__(self, in_dim, out_dim):
                super(LstmClf_, self).__init__()
                self.rnn = RNNStack(in_dim, hid_channels=128, residual=False,
                        preserve_len=True)
                self.lin = nn.Linear(128, output_dim)

            def forward(self, x, lens=None):
                x = self.rnn(x, lens)[0]
                x = self.lin(F.relu(x, inplace=True))
                return x.view(x.size(0), -1, 1, x.size(-1))

        self.predictors = nn.ModuleDict({
            'linear': nn.Sequential(
                # BW(HC)
                utils.Permute(0, 2, 1),
                # B(CH)W
                nn.Conv1d(input_dim, output_dim, kernel_size=aggreg,
                    padding=aggreg // 2),
                utils.Permute(0, 2, 1),
                # BW(HC)
                utils.Reshape(-1, 1, output_dim)
            ),
            'mlp': nn.Sequential(
                utils.Permute(0, 2, 1),
                nn.Conv1d(input_dim, 128, kernel_size=aggreg, padding=aggreg // 2),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, output_dim, 1),
                utils.Permute(0, 2, 1),
                utils.Reshape(-1, 1, output_dim)
            ),
            'lstm': LstmClf_(input_dim, output_dim)
        })

    def call_(self, pred, x, lens=None):
        if pred == 'lstm':
            return self.predictors['lstm'](x, lens)
        x = x.contiguous().view(x.size(0), x.size(1), -1)
        return self.predictors[pred](x)

    def forward(self, x, lens=None):
        preds = {
            fn_name: self.call_(fn_name, x, lens)
            for fn_name in self.predictors.keys()
                if self.only is None or fn_name in self.only
        }

        return preds



class TripleFramewisePredictor(nn.Module):
    """
    This auxiliary module is meant to be attached to a bottleneck on which you
    want to run some additional classifiers for ie monitoring what information
    is transmitted through it. Beware not to backprop it to the spine model.
    Args:
        input_dim: the size of the input features
        output_dim: the number of classes to discriminate against
        aggreg: how many consecutive frames to aggregate for prediction
    """

    def __init__(self, input_dim, output_dim, aggreg=3, ignore_padding=True,
            only=None):
        super(TripleFramewisePredictor, self).__init__()
        self.kernel_size = aggreg
        self.ignore_padding = ignore_padding
        self.pred = TripleClassifier(input_dim, output_dim, aggreg, only=only)

    def forward(self, vq_output, lens=None):
        return self.pred(vq_output, lens=lens)

    def plot(self, x, pred):
        import matplotlib.pyplot as plt

        fig = plt.Figure()
        (top, bottom) = fig.subplots(2)
        top.margins(y=0)
        bottom.margins(y=0)
        fig.tight_layout(h_pad=0)
        top.imshow(x[0, :, :, 0].cpu().transpose(0, 1))
        bottom.imshow(
            pred[0, :, 0, :].cpu().transpose(0, 1),
            aspect="auto",
            interpolation="nearest",
        )
        top.get_xaxis().set_visible(False)
        return fig

    def calculateFeatureLens(self, features, features_len):
        if features_len is None:
            features_len = torch.full(
                (features.shape[0],), fill_value=features.shape[1],
                device='cpu', dtype=torch.int64)
        return features_len

    def calculateInputLengths(self, input, features, features_len):
        feat_aligned_len = features.shape[1]
        hidden_aligned_len = input.shape[1]
        rate_factor = feat_aligned_len // hidden_aligned_len
        assert (feat_aligned_len % hidden_aligned_len) == 0, (
            "The hidden (captured) representation should evenly divide the "
            "features length"
        )
        input_lens = (features_len + rate_factor - 1) // rate_factor
        return input_lens, rate_factor

    def loss(self, features, targets, features_len=None, targets_len=None):
        # the features may be padded
        if features_len is None:
            assert targets_len is None
            assert features.shape[1] == targets.shape[1], (
                f"The lengths of the targets and the inputs should "
                f"be the same for a framewise prediction. "
                f"Currently: {targets.shape[1]} and {features.shape[1]} respectively."
            )
        else:
            assert (torch.all(features_len == targets_len) and
                    (features.shape[1] >= targets.shape[1]))
        features_len = self.calculateFeatureLens(features, features_len)
        inputs_len, rate_factor = self.calculateInputLengths(
            self.input, features, features_len)
        feat_aligned_len = features.shape[1]
        assert feat_aligned_len >= features_len.max(), (
            f"Incompatible shapes for features, pred, targets: "
            f"{(features.shape, pred.shape, targets.shape)}"
        )
        targets = targets.long()


        details = {}
        total_loss = 0
        for pred_name, pred in self(self.input, inputs_len).items():
            hidden_aligned_len = pred.shape[1]

            assert (feat_aligned_len % hidden_aligned_len) == 0, (
                "The hidden (captured) representation should evenly divide the "
                "features length"
            )
            pred = pred.repeat_interleave(rate_factor, dim=1)
            assert features_len.max() <= pred.shape[1], (
                f" Incompatible shapes for features_len, pred.shape[1]: "
                f"{(features_len.max(), pred.shape[1])}"
            )
            pred = pred[:, :targets.shape[1]].contiguous()

            pred_labels = utils.safe_squeeze(pred.argmax(dim=3), 2)
            accs = (pred_labels == targets).float()

            losses = F.cross_entropy(
                utils.safe_squeeze(pred, 2).permute(0, 2, 1), targets,
                reduction="none"
            )

            mask = utils.get_mask1d(features_len.to(losses.device), mask_length=losses.size(1))
            mask = mask / mask.sum()

            if not self.ignore_padding:
                mask[:] =1

            acc = (accs * mask).sum()
            loss = (losses * mask).sum()

            if logger.is_currently_logging():
                logger.log_mpl_figure(
                    "framewise_debug_" + pred_name,
                    self.plot(features, F.softmax(pred.detach(), dim=-1))
                )
            total_loss = total_loss + loss
            details.update({
                    "loss_" + pred_name: loss,
                    "acc_" + pred_name: acc,
                    "out_seq_" + pred_name: pred_labels.detach()})
        return total_loss, details

class FramewisePredictor(nn.Module):
    """
    This auxiliary module is meant to be attached to a bottleneck on which you
    want to run some additional classifiers for ie monitoring what information
    is transmitted through it. Beware not to backprop it to the spine model.

    Args:
        input_dim: the size of the input features
        output_dim: the number of classes to discriminate against
        aggreg: how many consecutive frames to aggregate for prediction
    """

    def __init__(self, input_dim, output_dim, aggreg=3,
            use_two_layer_predictor=False, ignore_padding=True):
        super(FramewisePredictor, self).__init__()
        self.kernel_size = aggreg
        self.ignore_padding = ignore_padding
        if use_two_layer_predictor:
            self.pred = nn.Sequential(
                nn.Conv1d(input_dim, 128, kernel_size=self.kernel_size),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, output_dim, 1),
            )
        else:
            self.pred = nn.Conv1d(input_dim, output_dim, kernel_size=self.kernel_size)

    def forward(self, vq_output):
        vq_output = (vq_output.contiguous()
                     .view(vq_output.size(0), vq_output.size(1), -1)
                     .permute(0, 2, 1))
        vq_output = F.pad(vq_output, (0, self.kernel_size - 1, 0, 0))
        out_conv = self.pred(vq_output)
        return out_conv.permute(0, 2, 1).unsqueeze(2)

    def plot(self, x, pred):
        import matplotlib.pyplot as plt

        fig = plt.Figure()
        (top, bottom) = fig.subplots(2)
        top.margins(y=0)
        bottom.margins(y=0)
        fig.tight_layout(h_pad=0)
        top.imshow(x[0, :, :, 0].cpu().transpose(0, 1))
        bottom.imshow(
            pred[0, :, 0, :].cpu().transpose(0, 1),
            aspect="auto",
            interpolation="nearest",
        )
        top.get_xaxis().set_visible(False)
        return fig

    def loss(self, features, targets, features_len=None, targets_len=None):
        # the features may be padded
        if features_len is None:
            assert targets_len is None
            assert features.shape[1] == targets.shape[1], (
                f"The lengths of the targets and the inputs should "
                f"be the same for a framewise prediction. "
                f"Currently: {targets.shape[1]} and {features.shape[1]} respectively."
            )
        else:
            assert (torch.all(features_len == targets_len) and
                    (features.shape[1] >= targets.shape[1]))
        lens = features_len

        if lens is None:
            lens = torch.full(
                (features.shape[0],), fill_value=features.shape[1],
                device=targets.device
            )

        hidden = self(self.input)
        feat_aligned_len = features.shape[1]
        hidden_aligned_len = hidden.shape[1]

        assert feat_aligned_len >= lens.max(), (
            f"Incompatible shapes for features, hidden, targets: "
            f"{(features.shape, hidden.shape, targets.shape)}"
        )
        targets = targets.long()

        rate_factor = feat_aligned_len // hidden_aligned_len
        assert (feat_aligned_len % hidden_aligned_len) == 0, (
            "The hidden (captured) representation should evenly divide the "
            "features length"
        )
        hidden = hidden.repeat_interleave(rate_factor, dim=1)
        assert lens.max() <= hidden.shape[1], (
            f" Incompatible shapes for lens, hidden.shape[1]: "
            f"{(lens.max(), hidden.shape[1])}"
        )
        hidden = hidden[:, :targets.shape[1]].contiguous()

        pred_labels = utils.safe_squeeze(hidden.argmax(dim=3), 2)
        accs = (pred_labels == targets).float()

        losses = F.cross_entropy(
            utils.safe_squeeze(hidden, 2).permute(0, 2, 1), targets,
            reduction="none"
        )

        mask = utils.get_mask1d(lens, mask_length=losses.size(1))
        mask = mask / mask.sum()

        if not self.ignore_padding:
            mask[:] = 1

        acc = (accs * mask).sum()
        loss = (losses * mask).sum()

        if logger.is_currently_logging():
            logger.log_mpl_figure(
                "framewise_debug",
                self.plot(features, F.softmax(hidden.detach(), dim=-1))
            )
        details = {"loss": loss, "acc": acc, "out_seq": pred_labels.detach()}
        return loss, details


class BaseCTCPredictor(nn.Module):
    """
    This auxiliary module is meant to be attached to a bottleneck on which you
    want to run some additional classifiers for ie monitoring what information
    is transmitted through it. Beware not to backprop it to the spine model.

    Args:
        input_dim: the size of the input features
        output_dim: the number of classes to discriminate against
        aggreg: how many consecutive frames to aggregate for prediction
    """

    def __init__(self, remove_reps_in_transcripts=True, zero_infinity=True,
                 loss_reduction='mean'):
        super(BaseCTCPredictor, self).__init__()
        reds = {'batch_mean': 'sum'}
        self.loss_reduction = loss_reduction
        self.ctc = torch.nn.CTCLoss(
            reduction=reds.get(loss_reduction, loss_reduction),
            zero_infinity=zero_infinity)
        self.remove_reps_in_transcripts = remove_reps_in_transcripts

    def forward(self, vq_output, features_len=None):
        raise NotImplementedError

    def plot(self, x, pred):
        import matplotlib.pyplot as plt

        fig = plt.Figure()
        (top, bottom) = fig.subplots(2)
        top.margins(y=0)
        bottom.margins(y=0)
        fig.tight_layout(h_pad=0)
        top.imshow(x[0, :, :, 0].cpu().transpose(0, 1))
        bottom.imshow(
            pred[:, 0, :].cpu().transpose(0, 1),
            aspect="auto", interpolation="nearest"
        )
        top.get_xaxis().set_visible(False)
        return fig

    def calculateFeatureLens(self, features, features_len):
        if features_len is None:
            features_len = torch.full(
                (features.shape[0],), fill_value=features.shape[1],
                device='cpu', dtype=torch.int64)
        return features_len

    def calculateInputLengths(self, input, features, features_len):
        feat_aligned_len = features.shape[1]
        hidden_aligned_len = input.shape[1]
        rate_factor = feat_aligned_len // hidden_aligned_len
        assert (feat_aligned_len % hidden_aligned_len) == 0, (
            "The hidden (captured) representation should evenly divide the "
            "features length"
        )
        input_lens = (features_len + rate_factor - 1) // rate_factor
        return input_lens, rate_factor

    def retrieve_saved_input(self):
        input = self.input  # get the tensor saved by the forward hook
        self.input = None
        return input

    def loss(self, features, targets, features_len=None, targets_len=None):
        features_len = self.calculateFeatureLens(features, features_len)
        input = self.retrieve_saved_input()
        # Calculate rate adjustments and lengths
        input_len, input_rate_factor = self.calculateInputLengths(
            input, features, features_len)

        if targets_len is not None:
            # Mask, and reduce batchsize to np.sum(mask)), if necessary.
            # We look at targets[] of zero length to determine
            # whether there are non-transcribed inputs.
            mask = targets_len > 0
            minTargetLen = min(targets_len).item()
            if minTargetLen == 0:
                input_len = input_len[mask]
                targets_len = targets_len[mask]
                features_len = features_len[mask]

                input = input[mask,:]
                targets = targets[mask,:]
                features = features[mask,:]

            # If empty batch after masking then return early with defaults
            if len(targets_len) == 0:
                loss = 0.
                cer = 100.0
                decodes = None
                details = {"loss": loss, "cer": torch.tensor(cer),
                           "out_seq": decodes}
                return loss, details

        # Forward CTCPredictor (on full input, features, before masking)
        hidden, hidden_len = self(input, features_len=input_len)

        # CTC target prep
        if self.remove_reps_in_transcripts:
            # Remove repetitions and blanks from the alignments
            new_targets = torch.zeros_like(targets).long()
            new_targets_len = torch.zeros_like(targets[:, 0])
            for i in range(targets.size(0)):
                tgt = targets[i].to("cpu").numpy()
                if targets_len is not None:
                    tgt = tgt[:targets_len[i]]
                # Let empty sequence map to a single blank
                tgt = utils.remove_reps_blanks(tgt) or [0]
                new_targets[i, :len(tgt)] = torch.tensor(tgt)
                new_targets_len[i] = len(tgt)
            targets = new_targets
            targets_len = new_targets_len

        targets_len = targets_len.long()
        # Workaround for issue #71
        targets = targets[:, : targets_len.max()].contiguous()

        # Log-prob preparation
        log_probs = hidden.permute(1, 0, 2)
        log_prob_lens = hidden_len
        log_probs = log_probs[:log_prob_lens.max()].contiguous() # see issue #71 equiv.

        assert (log_probs.size(0) == log_prob_lens[0]
                ), f"Not {log_probs.size(0)} == {log_prob_lens[0]}"
        assert not torch.isnan(log_probs).any()
        # Log probs are at encoder's rate while target are at input's rate
        assert not (targets_len > log_probs.size(0) * input_rate_factor).any()

        assert targets.max() <= log_probs.shape[-1]
        loss = self.ctc(log_probs, targets, log_prob_lens, targets_len)
        if self.loss_reduction == 'batch_mean':
            loss = loss / features.shape[0]

        decodes = utils.greedy_ctc_decode(log_probs, log_prob_lens)
        cer = utils.error_rate(
            decodes, [t[:tlen] for t, tlen in zip(targets.to("cpu"), targets_len)]
        )

        if logger.is_currently_logging():
            logger.log_mpl_figure(
                "ctc_predictor_debug", self.plot(features, torch.exp(log_probs.detach()))
            )
        details = {"loss": loss, "cer": torch.tensor(cer), "out_seq": decodes}
        return loss, details


class CTCPredictor(BaseCTCPredictor):
    """
    This auxiliary module is meant to be attached to a bottleneck on which you
    want to run some additional classifiers for ie monitoring what information
    is transmitted through it. Beware not to backprop it to the spine model.

    Args:
        input_dim: the size of the input features
        output_dim: the number of classes to discriminate against
        aggreg: how many consecutive frames to aggregate for prediction
    """

    def __init__(self, input_dim, output_dim, aggreg=3, **kwargs):
        super(CTCPredictor, self).__init__(**kwargs)
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=aggreg)
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(self, vq_output, features_len=None):
        vq_output = vq_output.contiguous().view(
            vq_output.size(0), vq_output.size(1), -1).permute(0, 2, 1)
        vq_output = F.pad(vq_output, (0, self.conv.kernel_size[0] - 1, 0, 0))
        out = self.conv(vq_output)
        out = out.permute(0, 2, 1)
        out = F.relu(out)
        out = self.fc(out)
        log_probs = F.log_softmax(out, dim=-1)
        return log_probs, features_len


class CTCPredictorLSTM(CTCPredictor):
    """This auxiliary module is meant to be attached to a bottleneck on which you
    want to run some additional classifiers for ie monitoring what information
    is transmitted through it. Beware not to backprop it to the spine model.

    This module is different from CTCPredictor that we actually align to targets (not alignments)
    and calculate a proper character error-rate (CER) to compare to supervised
    models. Only this head is trained to map embeddings to symbols. We choose
    a head that almost the same as the BLSTMs in the best supervised baseline, i.e,
    2 layers of BLSTMs. The classifier is Conv -> BLSTM->BLSTM->target

    Args:
        input_dim: the size of the input features
        output_dim: the number of classes to discriminate against
        aggreg: how many consecutive frames to aggregate for prediction
    """

    def __init__(self, input_dim, output_dim, hidden_dim=128, aggreg=3,
                 remove_reps_in_transcripts=True,
                 zero_infinity=True, loss_reduction='mean', rnn_args=None):

        # Pass hidden dim to the superclass to set the dimesnionality of the
        # Conv layer
        super(CTCPredictorLSTM, self).__init__(
            input_dim, output_dim=hidden_dim, aggreg=aggreg,
            remove_reps_in_transcripts=remove_reps_in_transcripts,
            zero_infinity=zero_infinity, loss_reduction=loss_reduction)

        # Setup the BLSTMs
        if rnn_args is None:
            rnn_args = dict(
                rnn_hidden_size=hidden_dim,
                rnn_nb_layers=2,
                rnn_projection_size=hidden_dim,
                rnn_type=nn.LSTM,
                rnn_dropout=False,
                rnn_residual=False,
                normalization='none',
                rnn_subsample=None,
                rnn_bidirectional=True,
                rnn_bias=True,
            )

        self.rnns, self.rnn_cumulative_stride = makeRnn(
            rnn_input_size=hidden_dim, **rnn_args
            )
        self.projectOutput = nn.Linear(hidden_dim, output_dim)

    def forwardBLSTM(self, in_, in_lens_):
        '''forwardBLSTM input is [batch x out_channels(=64) x seqLen]
            and output and after permute [batch x seqLen x nClasses]
        '''
        # Padding before conv
        out = F.pad(in_, (0, self.conv.kernel_size[0] - 1, 0, 0))
        out = self.conv(out)
        # assume conv did not change length/rate

        out = F.leaky_relu(out)

        # From [batchSize x nClasses x seqLen] to [SeqLen x batchSize x outChannels] to please LSTM
        out = out.permute(2, 0, 1)

        # Rnn pack + forward + unpack
        out = nn.utils.rnn.pack_padded_sequence(out, in_lens_.data.cpu().numpy())

        # in and out: (seqlen x bs x nClasses)
        out = self.rnns(out)

        # Outputs (bs x seqlen x nClasses)
        out, out_lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        return out, out_lens

    def forward(self, vq_output, features_len=None, targets_len=None):
        vq_output = (vq_output
                     .contiguous()
                     .view(vq_output.size(0), vq_output.size(1), -1)
                     .permute(0, 2, 1))
        # Run the RNN classifier
        out, out_lens = self.forwardBLSTM(vq_output, features_len)
        out = self.projectOutput(out)

        # Now we have output [batchSize x SeqLen x nChannels]
        log_probs = F.log_softmax(out, dim=-1)

        return log_probs, out_lens


class MappingCTCPredictor(BaseCTCPredictor):
    """
    Embeds token indices and trains a CTC predictor.
    Constructs a mapping from tokens to the target alphabet.
    Args:
        input_dim: the size of the input features
        output_dim: the number of classes to discriminate against
        num_tokens: number of tokens in the discrete bottleneck
        emb_size: token embedding size
    """
    def __init__(self, input_dim, output_dim, num_tokens=None, emb_size=128,
                 **kwargs):
        super(MappingCTCPredictor, self).__init__(**kwargs)
        del input_dim  # unused
        if num_tokens is None:
            raise ValueError(
                'MappingCTCPredictor works only with discrete bottlenecks')
        self.emb = nn.Embedding(num_tokens, emb_size)
        self.proj = nn.Linear(emb_size, output_dim)

    def retrieve_saved_input(self):
        # Pick 'indices' and squeeze to (bsz x L)
        indices = utils.safe_squeeze(self.input['indices'], -1)
        indices = utils.safe_squeeze(indices, -1)
        self.input = None
        return indices

    def forward(self, indices, features_len=None):
        emb = self.emb(indices)
        out = self.proj(emb)
        log_probs = F.log_softmax(out, dim=-1)
        return log_probs, features_len
