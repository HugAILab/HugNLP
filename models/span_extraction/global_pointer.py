# -*- coding: utf-8 -*-
# @Time    : 2022/4/21 5:30 下午
# @Author  : JianingWang
# @File    : global_pointer.py
from typing import Optional
import torch
import numpy as np
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import BCEWithLogitsLoss
from transformers import MegatronBertModel, MegatronBertPreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from roformer import RoFormerPreTrainedModel, RoFormerModel, RoFormerModel


class RawGlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size,
                               self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len,
                                    dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack(
            [torch.sin(embeddings),
             torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat(
            (batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings,
                                   (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask,
                                       token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]
        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(
                batch_size, seq_len, self.inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(
            batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim**0.5


class SinusoidalPositionEmbedding(nn.Module):
    """定义Sin-Cos位置Embedding."""
    def __init__(self,
                 output_dim,
                 merge_mode='add',
                 custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs, position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack(
            [torch.sin(embeddings),
             torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (
        1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    # print(y_pred, y_true, pos_loss)
    return (neg_loss + pos_loss).mean()


def multilabel_categorical_crossentropy2(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred.clone()
    y_pred_pos = y_pred.clone()
    y_pred_neg[y_true > 0] -= float('inf')
    y_pred_pos[y_true < 1] -= float('inf')
    # y_pred_neg = y_pred - y_true * float('inf')  # mask the pred outputs of pos classes
    # y_pred_pos = y_pred - (1 - y_true) * float('inf')  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    # print(y_pred, y_true, pos_loss)
    return (neg_loss + pos_loss).mean()


@dataclass
class GlobalPointerOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    topk_probs: torch.FloatTensor = None
    topk_indices: torch.IntTensor = None


class BertForEffiGlobalPointer(BertPreTrainedModel):
    def __init__(self, config):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super().__init__(config)
        self.bert = BertModel(config)
        self.ent_type_size = config.ent_type_size
        self.inner_dim = config.inner_dim
        self.hidden_size = config.hidden_size
        self.RoPE = config.RoPE

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(
            self.hidden_size, self.ent_type_size *
            2)  # 原版的dense2是(inner_dim * 2, ent_type_size * 2)

    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels=None,
                short_labels=None):
        # with torch.no_grad():
        context_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs.last_hidden_state  # [bz, seq_len, hidden_dim]
        outputs = self.dense_1(last_hidden_state)  # [bz, seq_len, 2*inner_dim]
        qw, kw = outputs[..., ::2], outputs[
            ..., 1::2]  # 从0,1开始间隔为2 最后一个纬度，从0开始，取奇数位置所有向量汇总
        batch_size = input_ids.shape[0]
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            cos_pos = pos[..., 1::2].repeat_interleave(
                2, dim=-1)  # e.g. [0.34, 0.90] -> [0.34, 0.34, 0.90, 0.90]
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim**0.5
        bias = torch.einsum('bnh->bhn', self.dense_2(last_hidden_state)) / 2
        logits = logits[:,
                        None] + bias[:, ::2,
                                     None] + bias[:, 1::2, :,
                                                  None]  # logits[:, None] 增加一个维度
        # logit_mask = self.add_mask_tril(logits, mask=attention_mask)
        loss = None

        mask = torch.triu(
            attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1))  # 上三角矩阵
        # mask = torch.where(mask > 0, 0.0, 1)
        if labels is not None:
            y_pred = logits - (1 - mask.unsqueeze(1)) * 1e12
            y_true = labels.view(input_ids.shape[0] * self.ent_type_size, -1)
            y_pred = y_pred.view(input_ids.shape[0] * self.ent_type_size, -1)
            loss = multilabel_categorical_crossentropy(y_pred, y_true)

        with torch.no_grad():
            prob = torch.sigmoid(logits) * mask.unsqueeze(1)
            topk = torch.topk(prob.view(batch_size, self.ent_type_size, -1),
                              50,
                              dim=-1)

        return GlobalPointerOutput(loss=loss,
                                   topk_probs=topk.values,
                                   topk_indices=topk.indices)


class RobertaForEffiGlobalPointer(RobertaPreTrainedModel):
    def __init__(self, config):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.ent_type_size = config.ent_type_size
        self.inner_dim = config.inner_dim
        self.hidden_size = config.hidden_size
        self.RoPE = config.RoPE

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(
            self.hidden_size, self.ent_type_size *
            2)  # 原版的dense2是(inner_dim * 2, ent_type_size * 2)

    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels=None,
                short_labels=None):
        # with torch.no_grad():
        context_outputs = self.roberta(input_ids, attention_mask,
                                       token_type_ids)
        last_hidden_state = context_outputs.last_hidden_state  # [bz, seq_len, hidden_dim]
        outputs = self.dense_1(last_hidden_state)  # [bz, seq_len, 2*inner_dim]
        qw, kw = outputs[..., ::2], outputs[
            ..., 1::2]  # 从0,1开始间隔为2 最后一个纬度，从0开始，取奇数位置所有向量汇总
        batch_size = input_ids.shape[0]
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            cos_pos = pos[..., 1::2].repeat_interleave(
                2, dim=-1)  # e.g. [0.34, 0.90] -> [0.34, 0.34, 0.90, 0.90]
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim**0.5
        bias = torch.einsum('bnh->bhn', self.dense_2(last_hidden_state)) / 2
        logits = logits[:,
                        None] + bias[:, ::2,
                                     None] + bias[:, 1::2, :,
                                                  None]  # logits[:, None] 增加一个维度
        # logit_mask = self.add_mask_tril(logits, mask=attention_mask)
        loss = None

        mask = torch.triu(
            attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1))  # 上三角矩阵
        # mask = torch.where(mask > 0, 0.0, 1)
        if labels is not None:
            y_pred = logits - (1 - mask.unsqueeze(1)) * 1e12
            y_true = labels.view(input_ids.shape[0] * self.ent_type_size, -1)
            y_pred = y_pred.view(input_ids.shape[0] * self.ent_type_size, -1)
            loss = multilabel_categorical_crossentropy(y_pred, y_true)

        with torch.no_grad():
            prob = torch.sigmoid(logits) * mask.unsqueeze(1)
            topk = torch.topk(prob.view(batch_size, self.ent_type_size, -1),
                              50,
                              dim=-1)

        return GlobalPointerOutput(loss=loss,
                                   topk_probs=topk.values,
                                   topk_indices=topk.indices)


class RoformerForEffiGlobalPointer(RoFormerPreTrainedModel):
    def __init__(self, config):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super().__init__(config)
        self.roformer = RoFormerModel(config)
        self.ent_type_size = config.ent_type_size
        self.inner_dim = config.inner_dim
        self.hidden_size = config.hidden_size
        self.RoPE = config.RoPE

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(
            self.hidden_size, self.ent_type_size *
            2)  # 原版的dense2是(inner_dim * 2, ent_type_size * 2)

    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels=None,
                short_labels=None):
        # with torch.no_grad():
        context_outputs = self.roformer(input_ids, attention_mask,
                                        token_type_ids)
        last_hidden_state = context_outputs.last_hidden_state  # [bz, seq_len, hidden_dim]
        outputs = self.dense_1(last_hidden_state)  # [bz, seq_len, 2*inner_dim]
        qw, kw = outputs[..., ::2], outputs[
            ..., 1::2]  # 从0,1开始间隔为2 最后一个纬度，从0开始，取奇数位置所有向量汇总
        batch_size = input_ids.shape[0]
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            cos_pos = pos[..., 1::2].repeat_interleave(
                2, dim=-1)  # e.g. [0.34, 0.90] -> [0.34, 0.34, 0.90, 0.90]
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim**0.5
        bias = torch.einsum('bnh->bhn', self.dense_2(last_hidden_state)) / 2
        logits = logits[:,
                        None] + bias[:, ::2,
                                     None] + bias[:, 1::2, :,
                                                  None]  # logits[:, None] 增加一个维度
        # logit_mask = self.add_mask_tril(logits, mask=attention_mask)
        loss = None

        mask = torch.triu(
            attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1))  # 上三角矩阵
        # mask = torch.where(mask > 0, 0.0, 1)
        if labels is not None:
            y_pred = logits - (1 - mask.unsqueeze(1)) * 1e12
            y_true = labels.view(input_ids.shape[0] * self.ent_type_size, -1)
            y_pred = y_pred.view(input_ids.shape[0] * self.ent_type_size, -1)
            loss = multilabel_categorical_crossentropy(y_pred, y_true)

        with torch.no_grad():
            prob = torch.sigmoid(logits) * mask.unsqueeze(1)
            topk = torch.topk(prob.view(batch_size, self.ent_type_size, -1),
                              50,
                              dim=-1)

        return GlobalPointerOutput(loss=loss,
                                   topk_probs=topk.values,
                                   topk_indices=topk.indices)


class MegatronForEffiGlobalPointer(MegatronBertPreTrainedModel):
    def __init__(self, config):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super().__init__(config)
        self.bert = MegatronBertModel(config)
        self.ent_type_size = config.ent_type_size
        self.inner_dim = config.inner_dim
        self.hidden_size = config.hidden_size
        self.RoPE = config.RoPE

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_dim * 2)
        self.dense_2 = nn.Linear(
            self.hidden_size, self.ent_type_size *
            2)  # 原版的dense2是(inner_dim * 2, ent_type_size * 2)

    def sequence_masking(self, x, mask, value='-inf', axis=None):
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'axis must be greater than 0'
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)
            return x * mask + value * (1 - mask)

    def add_mask_tril(self, logits, mask):
        if mask.dtype != logits.dtype:
            mask = mask.type(logits.dtype)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 2)
        logits = self.sequence_masking(logits, mask, '-inf', logits.ndim - 1)
        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), diagonal=-1)
        logits = logits - mask * 1e12
        return logits

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels=None,
                short_labels=None):
        # with torch.no_grad():
        context_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs.last_hidden_state  # [bz, seq_len, hidden_dim]
        outputs = self.dense_1(last_hidden_state)  # [bz, seq_len, 2*inner_dim]
        qw, kw = outputs[..., ::2], outputs[
            ..., 1::2]  # 从0,1开始间隔为2 最后一个纬度，从0开始，取奇数位置所有向量汇总
        batch_size = input_ids.shape[0]
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.inner_dim, 'zero')(outputs)
            cos_pos = pos[..., 1::2].repeat_interleave(
                2, dim=-1)  # e.g. [0.34, 0.90] -> [0.34, 0.34, 0.90, 0.90]
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.inner_dim**0.5
        bias = torch.einsum('bnh->bhn', self.dense_2(last_hidden_state)) / 2
        logits = logits[:,
                        None] + bias[:, ::2,
                                     None] + bias[:, 1::2, :,
                                                  None]  # logits[:, None] 增加一个维度
        # logit_mask = self.add_mask_tril(logits, mask=attention_mask)
        loss = None

        mask = torch.triu(
            attention_mask.unsqueeze(2) * attention_mask.unsqueeze(1))  # 上三角矩阵
        # mask = torch.where(mask > 0, 0.0, 1)
        if labels is not None:
            y_pred = logits - (1 - mask.unsqueeze(1)) * 1e12
            y_true = labels.view(input_ids.shape[0] * self.ent_type_size, -1)
            y_pred = y_pred.view(input_ids.shape[0] * self.ent_type_size, -1)
            loss = multilabel_categorical_crossentropy(y_pred, y_true)

        with torch.no_grad():
            prob = torch.sigmoid(logits) * mask.unsqueeze(1)
            topk = torch.topk(prob.view(batch_size, self.ent_type_size, -1),
                              50,
                              dim=-1)

        return GlobalPointerOutput(loss=loss,
                                   topk_probs=topk.values,
                                   topk_indices=topk.indices)
