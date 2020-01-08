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

from distsup import utils
from distsup.logger import default_tensor_logger
from distsup.modules import bert, convolutional, reconstructors, quantizers

logger = default_tensor_logger.DefaultTensorLogger()


class BaseMaskingAuxHead(nn.Module):
    def __init__(self,
                 image_height,
                 input_dim,
                 output_dim,
                 mask_len,
                 alphabet_size,
                 in_channels,
                 rand_mask_len=False,
                 hid_size=64,
                 num_attn_layers=2,
                 bert_hid_size=64,
                 bottleneck_latent_dim=64,
                 max_seq_len=128,
                 quantizer=dict(class_name=quantizers.L1Loss)):
        super(BaseMaskingAuxHead, self).__init__()
        del input_dim  # unused (aux_modules use it)
        del output_dim  # unused (aux_modules use it)
        self.hid_size = hid_size
        self.mask_len = mask_len
        self.max_seq_len = max_seq_len
        self.rand_mask_len = rand_mask_len
        self.quantizer = utils.construct_from_kwargs(quantizer)
        self.emb = nn.Conv1d(bottleneck_latent_dim, hid_size, kernel_size=1)
        self.input_conv = nn.Conv1d(image_height * in_channels, hid_size, 1, 1)

        self.positional = bert.PositionalEmbedding(hid_size)
        self.logged_sample = False

        bert_config = bert.BertConfig(hidden_size=bert_hid_size,
                                      num_attention_heads=8,
                                      intermediate_size=4*self.hid_size)
        self.attn_stack = nn.ModuleList()
        for _ in range(num_attn_layers):
            self.attn_stack.append(bert.BertLayer(bert_config))

    def get_lens(self, features, targets, features_len, targets_len):
        if features_len is None:
            assert targets_len is None
            assert features.shape[1] == targets.shape[1]
            return torch.full(
                (features.shape[0],), fill_value=features.shape[1],
                device=targets.device)
        else:
            assert (torch.all(features_len == targets_len) and
                    (features.shape[1] >= targets.shape[1]))
            return features_len

    def forward(self, vq_output, features_len=None):
        return vq_output, features_len

    def maybe_log_sample(self, masked, reconstructed):
        if self.training:
            self.logged_sample = False
            return
        elif not self.logged_sample:
            def log_img(name, img):
                # Convert BTH to  HWC (W becomes B*T, C=1)
                assert len(img.size()) == 3
                img = img.contiguous().view(-1, img.size(2))
                img = img.permute(1, 0).unsqueeze(-1)  # To HWC
                logger.log_image(name, img)
            log_img('reconstructed', reconstructed)
            log_img('masked', masked)
            self.logged_sample = True

    def get_saved_inputs(self, features, x_lens):
        # Retrieve bottleneck tokens saved by the fwd hook
        hidden = self.input
        self.input = None
        feat_aligned_len = features.shape[1]
        hidden_aligned_len = hidden.shape[1]

        assert feat_aligned_len >= x_lens.max()
        rate_factor = feat_aligned_len // hidden_aligned_len
        assert (feat_aligned_len % hidden_aligned_len) == 0
        return hidden, rate_factor

    def _forward(self, x, x_lens, cond=None):
        # Extract the masked part of the input
        assert self.mask_len % 2 == 1
        if self.rand_mask_len:
            mask_len2 = np.random.randint(0, (self.mask_len + 1) // 2)
            mask_len = mask_len2 * 2 + 1
        else:
            mask_len = self.mask_len
            mask_len2 = mask_len // 2
        mask_pos = np.random.randint(0, x_lens.min() - mask_len)

        masked = x[:, mask_pos:mask_pos + mask_len].permute(0, 1, 3, 2)
        # masked: (bsz x mask_len x h x c)

        input_masked = x.clone()
        input_masked[:, mask_pos:mask_pos + mask_len] = 0.0

        # x: (bsz x t x c x h)
        N, W, C, H = x.size()
        x = x.view(N, W, C * H).permute(0, 2, 1)
        x = self.input_conv(x)
        x = x.permute(0, 2, 1)
        # x: (bsz x t x dim)

        # `ind_seqs` determine the order of inputs for every row;
        # these should be cnetered in the middle of the mask with value 0
        pos_emb_arange = torch.arange(-mask_pos - mask_len2,
                                      -mask_pos - mask_len2 + x_lens[0],
                                      dtype=torch.float32,
                                      device=x_lens.device)
        pos_emb = self.positional(pos_emb_arange, bsz=x.size(0))
        # pos_emb -> (bsz x t x dim)
        pos_emb = pos_emb.permute(1, 0, 2).contiguous()

        # Apply masking
        x[:, mask_pos:mask_pos + mask_len] = 0.0

        x = x + pos_emb

        if cond is not None:
            assert x.size(1) == cond.size(1)
            # (bs x t) -> (bs x t x dim)
            cond = cond + pos_emb
            x = torch.cat([x, cond], dim=2)

        # mask: (t x 1 x 1 x bsz) ; allow broadcasting
        # Empty mask (additive) for compatibility with Bert module
        # - we're not using it, since we stack masked x with unmasked z
        attn_mask = torch.zeros(x.size(0), 1, x.size(1), 1,
                                dtype=x.dtype, device=x.device)
        # x: (bsz x t x 2dim)
        # x = x.contiguous()
        for attn_ff in self.attn_stack:
            x = attn_ff(x, attn_mask)
            x = x[0] if type(x) is tuple and len(x) == 1 else x

        # XXX Decoder_2d is fragile, disabled for now (use with pixshuff=False)
        if False or type(self.reconstructor) is reconstructors.Decoder_2d:
            rec = self.reconstructor(input_masked, (x.unsqueeze(2),))
            # rec:(bsz x t x h x c x 1) -> (bsz x t x h x c)
            rec = rec.squeeze(-1).contiguous()
        else:
            rec = self.reconstructor(x)
        # rec: (bsz x t x h x c)
        return rec, masked, mask_pos, mask_len


class MaskingAuxHead(BaseMaskingAuxHead):
    """Reconstructs masked img using unmasked parts of img and latents"""
    def __init__(self,
                 image_height,
                 input_dim,
                 output_dim,
                 mask_len,
                 alphabet_size,
                 hid_size=64,
                 in_channels=1,
                 reconstructor=dict(
                    class_name=reconstructors.DownsamplingDecoder2D,
                    num_layers=6,
                 ),
                 **kwargs):
        super(MaskingAuxHead, self).__init__(
            image_height=image_height, input_dim=input_dim,
            output_dim=output_dim, mask_len=mask_len,
            in_channels=in_channels, hid_size=hid_size, bert_hid_size=2*hid_size,
            alphabet_size=alphabet_size, **kwargs)

        rec_add_params = {'image_height': image_height,
                           'input_dim': 2*self.hid_size}
        # XXX Disabled for now
        if False and reconstructor['class_name'].endswith('.Decoder_2d'):
            del rec_add_params['input_dim']
            rec_add_params.update({
                'out_channels' : in_channels,
                'cond_channels': [{
                    'cond_dim': 2*self.hid_size,
                    'reduction_factor': 1}]})
        self.reconstructor = utils.construct_from_kwargs(
            reconstructor, additional_parameters=rec_add_params)

    def loss(self, features, targets, features_len, targets_len):

        x = features
        x_lens = self.get_lens(features, targets, features_len, targets_len)
        z, rate_factor = self.get_saved_inputs(features, x_lens)

        # Upscale tokens to match the input signal
        z = z.repeat_interleave(rate_factor, dim=1)
        assert x_lens.max() <= z.shape[1]
        # x: (bsz x t x 1 x h)
        z = z[:, :targets.shape[1]].contiguous()
        x = x.permute(0, 1, 3, 2)
        z = self.emb(z.squeeze(2).permute(0, 2, 1)).permute(0, 2, 1)
        # x: (bsz x t x h x c)
        x_lens = x_lens.long()

        # Self attn over long sequences won't fit in mem
        if x_lens.max() > self.max_seq_len:
            x_lens = torch.min(torch.tensor([self.max_seq_len]).to(x_lens),
                               x_lens)
        x = x[:, :x_lens[0]]
        z = z[:, :x_lens[0]]

        rec, masked, mask_pos, mask_len = self._forward(x, x_lens, cond=z)
        rec = rec[:, mask_pos:mask_pos + mask_len].contiguous()
        if rec.size(3) == 1:  # single-channel image
            self.maybe_log_sample(utils.safe_squeeze(masked, -1),
                                  utils.safe_squeeze(rec, -1))

        loss = self.quantizer.loss(rec, utils.safe_squeeze(masked, -1)).mean()
        return loss, {'loss': loss}


class MaskingLatentsAuxHead(BaseMaskingAuxHead):
    """Reconstructs masked latents using other unmasked latents"""
    def __init__(self,
                 image_height,
                 input_dim,
                 output_dim,
                 mask_len,
                 alphabet_size,
                 hid_size=64,
                 **kwargs):
        super(MaskingLatentsAuxHead, self).__init__(
            image_height=image_height, input_dim=input_dim,
            output_dim=output_dim, mask_len=mask_len,
            hid_size=hid_size, bert_hid_size=hid_size,
            alphabet_size=alphabet_size, **kwargs)

    def loss(self, features, targets, features_len, targets_len):

        lens = self.get_lens(features, targets, features_len, targets_len)
        z, rate_factor = self.get_saved_inputs(features, x_lens)

        lens = (lens // rate_factor).long()
        assert lens.min() >= 1
        z = z[:, :lens[0].item()].contiguous()
        z = z.squeeze(2)
        z = self.emb(z.permute(0, 2, 1)).permute(0, 2, 1)

        logits, masked, starts = self._forward(z, lens, cond=None)

        rec, masked, mask_pos, mask_len = self._forward(z, lens, cond=None)
        rec = rec[:, mask_pos:mask_pos + mask_len].contiguous()

        self.maybe_log_sample(masked, rec)

        # masked: (bsz x t x h x 1) -> (bsz x t x h)
        loss = self.quantizer.loss(rec, utils.safe_squeeze(masked, -1)).mean()
        return loss, {'loss': loss}
