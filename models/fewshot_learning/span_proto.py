# -*- coding: utf-8 -*-
# @Time    : 2022/4/21 5:30 下午
# @Author  : JianingWang
# @File    : span_proto.py
"""This code is implemented for the paper ''SpanProto: A Two-stage Span-based Prototypical Network for Few-shot Named Entity Recognition''."""

import os
from typing import Optional
import torch
import numpy as np
import torch.nn as nn
from typing import Union
from dataclasses import dataclass
from torch.nn import BCEWithLogitsLoss
from transformers import MegatronBertModel, MegatronBertPreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.models.bert import BertPreTrainedModel, BertModel

a = torch.nn.Embedding(10, 20)
a.parameters


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
    last_hidden_state: torch.FloatTensor = None


@dataclass
class SpanProtoOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    query_spans: list = None
    proto_logits: list = None
    topk_probs: torch.FloatTensor = None
    topk_indices: torch.IntTensor = None


class SpanDetector(BertPreTrainedModel):
    def __init__(self, config):
        # encodr: RoBerta-Large as encoder
        # inner_dim: 64
        # ent_type_size: ent_cls_num
        super().__init__(config)
        self.bert = BertModel(config)
        # self.ent_type_size = config.ent_type_size
        self.ent_type_size = 1
        self.inner_dim = 64
        self.hidden_size = config.hidden_size
        self.RoPE = True

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
        del context_outputs
        outputs = self.dense_1(last_hidden_state)  # [bz, seq_len, 2*inner_dim]
        qw, kw = outputs[..., ::2], outputs[
            ..., 1::2]  # 从0,1开始间隔为2 最后一个维度，从0开始，取奇数位置所有向量汇总
        batch_size = input_ids.shape[0]
        if self.RoPE:  # 是否使用RoPE旋转位置编码
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
            # y_pred = torch.zeros(input_ids.shape[0], self.ent_type_size, input_ids.shape[1], input_ids.shape[1], device=input_ids.device)
            # for i in range(input_ids.shape[0]):
            #     for j in range(self.ent_type_size):
            #         y_pred[i, j, labels[i, j, 0], labels[i, j, 1]] = 1
            # y_true = labels.reshape(input_ids.shape[0] * self.ent_type_size, -1)
            # y_pred = logit_mask.reshape(input_ids.shape[0] * self.ent_type_size, -1)
            # loss = multilabel_categorical_crossentropy(y_pred, y_true)
            #

            # weight = ((labels == 0).sum() / labels.sum())/5
            # loss_fct = nn.BCEWithLogitsLoss(weight=weight)
            # loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            # unmask_labels = labels.view(-1)[mask.view(-1) > 0]
            # loss = loss_fct(logits.view(-1)[mask.view(-1) > 0], unmask_labels.float())
            # if unmask_labels.sum() > 0:
            #     loss = (loss[unmask_labels > 0].mean()+loss[unmask_labels < 1].mean())/2
            # else:
            #     loss = loss[unmask_labels < 1].mean()
            # y_pred = logits.view(-1)[mask.view(-1) > 0]
            # y_true = labels.view(-1)[mask.view(-1) > 0]
            # loss = multilabel_categorical_crossentropy2(y_pred, y_true)
            # y_pred = logits - torch.where(mask > 0, 0.0, float('inf')).unsqueeze(1)
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
                                   topk_indices=topk.indices,
                                   last_hidden_state=last_hidden_state)


class SpanProto(nn.Module):
    def __init__(self, config):
        '''
        word_encoder: Sentence encoder

        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.config = config
        self.output_dir = './outputs'
        # self.predict_dir = self.predict_result_path(self.output_dir)
        self.drop = nn.Dropout()
        self.global_span_detector = SpanDetector(
            config=self.config)  # global span detector
        self.projector = nn.Sequential(  # projector
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Sigmoid(),
            # nn.LayerNorm(2)
        )
        self.tag_embeddings = nn.Embedding(
            2, self.config.hidden_size)  # tag for labeled / unlabeled span set
        # self.tag_mlp = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.max_length = 64
        self.margin_distance = 6.0
        self.global_step = 0

    def predict_result_path(self, path=None):
        if path is None:
            predict_dir = os.path.join(
                self.output_dir, '{}-{}-{}'.format(self.mode, self.num_class,
                                                   self.num_example),
                'predict')
        else:
            predict_dir = os.path.join(path, 'predict')
        # if os.path.exists(predict_dir):
        #     os.rmdir(predict_dir) # 删除历史记录
        if not os.path.exists(predict_dir):  # 重新创建一个新的目录
            os.makedirs(predict_dir)
        return predict_dir

    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path: Optional[Union[str,
                                                               os.PathLike]],
            *model_args, **kwargs):
        config = kwargs.pop('config', None)
        model = SpanProto(config=config)
        # 将bert部分参数加载进去
        model.global_span_detector = SpanDetector.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs)
        # 将剩余的参数加载进来
        return model

    # @classmethod
    # def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
    #     self.global_span_detector.resize_token_embeddings(new_num_tokens)

    def __dist__(self, x, y, dim, use_dot=False):
        # x: [1, class_num, hidden_dim], y: [span_num, 1, hidden_dim]
        # x - y: [span_num, class_num, hidden_dim]
        # (x - y)^2.sum(2): [span_num, class_num]
        if use_dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __get_proto__(self,
                      support_emb: torch,
                      support_span: list,
                      support_span_type: list,
                      use_tag=False):
        '''
        support_emb: [n', seq_len, dim]
        support_span: [n', m, 2] e.g. [[[3, 6], [12, 13]], [[1, 3]], ...]
        support_span_type: [n', m] e.g. [[2, 1], [5], ...]
        '''
        prototype = list()  # 每个类的proto type
        all_span_embs = list()  # 保存每个span的embedding
        all_span_tags = list()
        # 遍历每个类
        for tag in range(self.num_class):
            # tag_id = torch.Tensor([1 if tag == self.num_class else 0]).long().cuda()
            # tag_embeddings = self.tag_embeddings(tag_id).view(-1)
            tag_prototype = list()  # [k, dim]
            # 遍历当前episode内的每个句子
            for emb, span, type in zip(support_emb, support_span,
                                       support_span_type):
                # emb: [seq_len, dim], span: [m, 2], type: [m]
                span = torch.Tensor(
                    span).long().cuda()  # e.g. [[3, 4], [9, 11]]
                type = torch.Tensor(type).long().cuda()  # e.g. [1, 4]
                # 获取当前句子中属于tag类的span
                try:
                    tag_span = span[type == tag]  # e.g. span==[[3, 4]], tag==1

                    # 遍历每个检索到的span，获得其span embedding
                    for (s, e) in tag_span:
                        # tag_emb = torch.cat([emb[s], emb[e - 1]]) # [2*dim]
                        tag_emb = emb[s] + emb[e]  # [dim]
                        # if use_tag: # 添加是否为unlabeled的标记，0对应embedding表示当前的span是labeled span，否则为unlabeled span
                        #     tag_emb = tag_emb + tag_embeddings
                        tag_prototype.append(tag_emb)
                        all_span_embs.append(tag_emb)
                        all_span_tags.append(tag)
                except:
                    # 说明当前类不存在对应的span，则随机
                    tag_prototype.append(
                        torch.randn(support_emb.shape[-1]).cuda())
                    # assert 1 > 2
            try:
                prototype.append(torch.mean(torch.stack(tag_prototype), dim=0))
            except:
                # print("the class {} has no span".format(tag))
                prototype.append(torch.randn(support_emb.shape[-1]).cuda())
                # assert 1 > 2
        all_span_embs = torch.stack(
            all_span_embs).detach().cpu().numpy().tolist()

        return torch.stack(
            prototype), all_span_embs, all_span_tags  # [num_class + 1, dim]

    def __batch_dist__(self, prototype: torch, query_emb: torch,
                       query_spans: list, query_span_type: Union[list, None]):
        """该函数用于获得query到各个prototype的分类."""
        # 首先获得当前episode的每个句子的每个span的表征向量
        # 遍历每个句子
        all_logits = list()  # 保存每个episode，每个句子所有span的预测概率
        all_types = list()
        visual_all_types, visual_all_embs = list(), list()  # 用于展示可视化
        # num = 0
        for emb, span in zip(query_emb, query_spans):  # 遍历每个句子
            # assert len(span) == len(query_span_type[num]), "span={}\ntype{}".format(span, query_span_type[num])
            # print("len(span)={}, len(type)= {}".format(len(span), len(query_span_type[num])))
            span_emb = list()  # 保存当前句子所有span的embedding [m', dim]
            try:
                for (s, e) in span:  # 遍历每个span
                    tag_emb = emb[s] + emb[e]  # [dim]
                    span_emb.append(tag_emb)
            except:
                span_emb = []
            if len(span_emb) != 0:
                span_emb = torch.stack(span_emb)  # [span_num, dim]
                # 每个span与prototype计算距离
                logits = self.__dist__(prototype.unsqueeze(0),
                                       span_emb.unsqueeze(1),
                                       2)  # [span_num, num_class]
                # pred_types = torch.argmax(logits, -1).detach().cpu().numpy().tolist()
                with torch.no_grad():
                    pred_dist, pred_types = torch.max(
                        logits, -1)  # 获得每个query与所有prototype的距离的最近的类及其距离的平方
                    pred_dist = torch.pow(-1 * pred_dist, 0.5)
                    # print("pred_dist=", pred_dist)
                    # 如果最近的距离超过了margin distant，则该span视为unlabeled span，标注为特殊的类
                    pred_types[
                        pred_dist > self.margin_distance] = self.num_class
                    pred_types = pred_types.detach().cpu().numpy().tolist()
                # # 获得概率分布
                # with torch.no_grad():
                #     prob = torch.softmax(logits, -1)
                #     pred_proba, pred_types = torch.max(logits, -1)  # 获得每个span预测概率最大的类及其概率
                #     pred_types[pred_proba <= 0.6] = self.num_class # 如果当前预测的最大概率不满足，则说明其可能是一个其他实体
                #     pred_types = pred_types.detach().cpu().numpy().tolist()

                all_logits.append(logits)
                all_types.append(pred_types)
                visual_all_types.extend(pred_types)
                visual_all_embs.extend(
                    span_emb.detach().cpu().numpy().tolist())
            else:
                all_logits.append([])
                all_types.append([])
            # num += 1

        if query_span_type is not None:
            # query_span_type: [n', m]
            try:
                all_type = torch.Tensor([
                    type for types in query_span_type for type in types
                ]).long().cuda()  # [span_num]
                loss = nn.CrossEntropyLoss()(torch.cat(all_logits, 0),
                                             all_type)
            except:
                all_logit, all_type = list(), list()
                for logits, types in zip(all_logits, query_span_type):
                    if len(logits) != 0 and len(types) != 0 and len(
                            logits) == len(types):
                        # print("len(logits)=", len(logits))
                        # print("len(types)=", len(types))
                        # print("logits=", logits)
                        all_logit.append(logits)
                        all_type.extend(types)
                # print("all_logit=", all_logit)
                if len(all_logit) != 0:
                    all_logit = torch.cat(all_logit, 0)
                    all_type = torch.Tensor(all_type).long().cuda()
                    # print("len(all_logits)=", len(all_logits))
                    # print("len(query_span_type)=", len(query_span_type))

                    # print("types.shape=", torch.Tensor(all_type).shape)

                    # min_len = min(len(all_type), len(all_type))
                    # all_logit, all_type = all_logit[: min_len], all_type[: min_len]
                    # print("logits.shape=", all_logit.shape)
                    # print('all_type=', all_type)
                    loss = nn.CrossEntropyLoss()(all_logit, all_type)
                else:
                    loss = 0.

        else:
            loss = None
        all_logits = [
            i.detach().cpu().numpy().tolist() for i in all_logits
            if len(i) != 0
        ]
        return loss, all_logits, all_types, visual_all_types, visual_all_embs

    def __batch_margin__(self, prototype: torch, query_emb: torch,
                         query_unlabeled_spans: list,
                         query_labeled_spans: list, query_span_type: list):
        """该函数用于拉开unlabeled span与各个prototype的距离，拉近labeled span到对应类别的距离."""

        # prototype: [num_class, dim], negative: [span_num, dim]
        # 获得每个unlabeled span与每个prototype的距离的平方，目标是对于每个距离平方都要设置大于margin阈值
        def distance(input1, input2, p=2, eps=1e-6):
            # Compute the distance (p-norm)
            norm = torch.pow(torch.abs((input1 - input2 + eps)), p)
            pnorm = torch.pow(torch.sum(norm, -1), 1.0 / p)
            return pnorm

        unlabeled_span_emb, labeled_span_emb, labeled_span_type = list(), list(
        ), list()
        for emb, span in zip(query_emb, query_unlabeled_spans):  # 遍历每个句子
            # 保存当前句子所有span的embedding [m', dim]
            for (s, e) in span:  # 遍历每个span
                tag_emb = emb[s] + emb[e]  # [dim]
                unlabeled_span_emb.append(tag_emb)

        # for emb, span, type in zip(query_emb, query_labeled_spans, query_span_type): # 遍历每个句子
        #       # 保存当前句子所有span的embedding [m', dim]
        #     for (s, e) in span: # 遍历每个span
        #         tag_emb = emb[s] + emb[e]  # [dim]
        #         labeled_span_emb.append(tag_emb)
        #     labeled_span_type.extend(type)

        try:
            unlabeled_span_emb = torch.stack(
                unlabeled_span_emb)  # [span_num, dim]
            # labeled_span_emb = torch.stack(labeled_span_emb) # [span_num, dim]
            # labeled_span_type = torch.stack(labeled_span_type) # [span_num]
        except:
            return 0.

        unlabeled_dist = distance(
            prototype.unsqueeze(0),
            unlabeled_span_emb.unsqueeze(1))  # [span_num, num_class]
        # labeled_dist = distance(prototype.unsqueeze(0), labeled_span_emb.unsqueeze(1)) # [span_num, num_class]
        # 获得每个span对应ground truth类别距离prototype的距离
        # labeled_type_dist = torch.gather(labeled_dist, -1, labeled_span_type.unsqueeze(1)) # [span_num, 1]
        # print(dist)
        unlabeled_output = torch.maximum(torch.zeros_like(unlabeled_dist),
                                         self.margin_distance - unlabeled_dist)
        # labeled_output = torch.maximum(torch.zeros_like(labeled_type_dist), labeled_type_dist)
        # return torch.mean(unlabeled_output) + torch.mean(labeled_output)
        return torch.mean(unlabeled_output)

    def forward(self,
                episode_ids,
                support,
                query,
                num_class,
                num_example,
                mode=None,
                short_labels=None,
                stage: str = 'train',
                path: str = None):
        '''
        episode_ids: Input of the idx of each episode data. (only list)
        support: Inputs of the support set.
        query: Inputs of the query set.
        num_class: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        if stage.startswith('train'):
            self.global_step += 1
        self.num_class = num_class  # N-way K-shot里的N
        self.num_example = num_example  # N-way K-shot里的K
        # print('num_class=', num_class)
        self.mode = mode  # FewNERD mode=inter/intra
        self.max_length = support['input_ids'].shape[1]
        support_inputs, support_attention_masks, support_type_ids = \
            support['input_ids'], support['attention_mask'], support['token_type_ids'] # torch, [n, seq_len]
        query_inputs, query_attention_masks, query_type_ids = \
            query['input_ids'], query['attention_mask'], query['token_type_ids'] # torch, [n, seq_len]
        support_labels = support['labels']  # torch,
        query_labels = query['labels']  # torch,
        # global span detector: obtain all mention span and loss
        support_detector_outputs = self.global_span_detector(
            support_inputs,
            support_attention_masks,
            support_type_ids,
            support_labels,
            short_labels=short_labels)
        query_detector_outputs = self.global_span_detector(
            query_inputs,
            query_attention_masks,
            query_type_ids,
            query_labels,
            short_labels=short_labels)
        device_id = support_inputs.device.index

        # if stage == "train_span":
        if self.global_step <= 500 and stage == 'train':
            # only train span detector
            return SpanProtoOutput(
                loss=support_detector_outputs.loss,
                topk_probs=query_detector_outputs.topk_probs,
                topk_indices=query_detector_outputs.topk_indices,
            )
        # obtain labeled span from the support set
        support_labeled_spans = support[
            'labeled_spans']  # all labeled span, list, [n, m, 2], n sentence, m entity span, 2 (start / end)
        support_labeled_types = support[
            'labeled_types']  # all labeled ent type id, list, [n, m],
        query_labeled_spans = query[
            'labeled_spans']  # all labeled span, list, [n, m, 2], n sentence, m entity span, 2 (start / end)
        query_labeled_types = query[
            'labeled_types']  # all labeled ent type id, list, [n, m],

        # for span, type in zip(query_labeled_spans, query_labeled_types): # 遍历每个句子
        #     assert len(span) == len(type), "span={}\ntype{}".format(span, type)

        # obtain unlabeled span from the support set
        # according to the detector, we can obtain multiple unlabeled span, which generated by the detector
        # but not labeled in n-way k-shot episode
        # support_predict_spans = self.get_topk_spans( #
        #     support_detector_outputs.topk_probs,
        #     support_detector_outputs.topk_indices,
        #     support['input_ids']
        # ) # [n, m, 2]
        # print('predicted support span num={}'.format([len(i) for i in support_predict_spans]))
        # e.g. 打印一个所有句子，每个元素表示每个句子中的span个数，[5, 50, 4, 43, 5, 5, 1, 50, 2, 5, 6, 4, 50, 8, 12, 28, 17]

        # we can also obtain all predicted span from the query set
        query_predict_spans = self.get_topk_spans(  #
            query_detector_outputs.topk_probs,
            query_detector_outputs.topk_indices,
            query['input_ids'],
            threshold=0.9 if stage.startswith('train') else 0.95,
            is_query=True)  # [n, m, 2]
        # print('predicted query span num={}'.format([len(i) for i in query_predict_spans]))

        # merge predicted span and labeled span, and generate other class for unlabeled span set
        # support_all_spans, support_span_types = self.merge_span(
        #     labeled_spans=support_labeled_spans,
        #     labeled_types=support_labeled_types,
        #     predict_spans=support_predict_spans,
        #     stage=stage
        # ) # [n, m, 2] n 个句子，每个句子有若干个span
        # print('merged support span num={}'.format([len(i) for i in support_all_spans]))

        if stage.startswith('train'):
            # 在训练阶段，需要知道detector识别的所有区间中，哪些是labeled，哪些是unlabeled，将unlabeled span全部分离出来
            query_unlabeled_spans = self.split_span(  # 拆分出unlabeled span，用于后面的margin loss
                labeled_spans=query_labeled_spans,
                labeled_types=query_labeled_types,
                predict_spans=query_predict_spans,
                stage=stage)  # [n, m, 2] n 个句子，每个句子有若干个span
            # print('merged query span num={}'.format([len(i) for i in query_all_spans]))
            query_all_spans = query_labeled_spans
            query_span_types = query_labeled_types

        else:
            # 在推理阶段，直接全部merge
            query_unlabeled_spans = None
            query_all_spans, _ = self.merge_span(
                labeled_spans=query_labeled_spans,
                labeled_types=query_labeled_types,
                predict_spans=query_predict_spans,
                stage=stage)  # [n, m, 2] n 个句子，每个句子有若干个span
            # 在dev和test时，此时query部分的span完全靠detector识别
            # query_all_spans = query_predict_spans
            query_span_types = None
            # 用于查看推理阶段dev或test的query上detector的预测结果
            # for query_label, query_pred in zip(query_labeled_spans, query_predict_spans):
            #     print(" ==== ")
            #     print('query_labeled_spans=', query_label)
            #     print('query_predict_spans=', query_pred)

        # obtain representations of each token
        support_emb, query_emb = support_detector_outputs.last_hidden_state, \
                                 query_detector_outputs.last_hidden_state # [n, seq_len, dim]
        support_emb, query_emb = self.projector(support_emb), self.projector(
            query_emb)  # [n, seq_len, dim]

        # all_query_spans = list() # 保存每个episode的所有句子所有的预测span
        # all_proto_logits = list() # 保存每个episode的所有句子每个预测span对应的entity type
        batch_result = dict()
        proto_losses = list()  # 保存每个episode的loss
        # batch_visual = list() # 保存每个episode所有span的表征向量，用于可视化
        current_support_num = 0
        current_query_num = 0
        typing_loss = None
        # 遍历每个episode
        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]
            id_ = episode_ids[i]  # 当前episode的编号

            # 对于support，只对labeled span获得prototype
            # locate one episode and obtain the span prototype
            # [n', seq_len, dim] n' sentence in one episode
            # support_proto [num_class + 1, dim]
            support_proto, all_span_embs, all_span_tags = self.__get_proto__(
                support_emb[current_support_num:current_support_num +
                            sent_support_num],  # [n', seq_len, dim]
                support_labeled_spans[current_support_num:current_support_num +
                                      sent_support_num],  # [n', m]
                support_labeled_types[current_support_num:current_support_num +
                                      sent_support_num],  # [n', m]
            )

            # 对于query set每个labeled span，使用标准的prototype learning
            # for each query, we first obtain corresponding span, and then calculate distance between it and each prototype
            # # [n', seq_len, dim] n' sentence in one episode
            proto_loss, proto_logits, all_types, visual_all_types, visual_all_embs = self.__batch_dist__(
                support_proto,
                query_emb[current_query_num:current_query_num +
                          sent_query_num],  # [n', seq_len, dim]
                query_all_spans[current_query_num:current_query_num +
                                sent_query_num],  # [n', m]
                query_span_types[current_query_num:current_query_num +
                                 sent_query_num]
                if query_span_types else None,  # [n', m]
            )

            visual_data = {
                'data': all_span_embs + visual_all_embs,
                'target': all_span_tags + visual_all_types,
            }

            # 对于query unlabeled span，遍历每个span，拉开与所有prototype的距离，选择margin loss
            if stage.startswith('train'):

                margin_loss = self.__batch_margin__(
                    support_proto,
                    query_emb[current_query_num:current_query_num +
                              sent_query_num],  # [n', seq_len, dim]
                    query_unlabeled_spans[current_query_num:current_query_num +
                                          sent_query_num],  # [n', span_num]
                    query_all_spans[current_query_num:current_query_num +
                                    sent_query_num],
                    query_span_types[current_query_num:current_query_num +
                                     sent_query_num],
                )

                proto_losses.append(proto_loss + margin_loss)

            batch_result[id_] = {
                'spans':
                query_all_spans[current_query_num:current_query_num +
                                sent_query_num],
                'types':
                all_types,
                'visualization':
                visual_data
            }

            current_query_num += sent_query_num
            current_support_num += sent_support_num
        # proto_logits = torch.stack(proto_logits)
        if stage.startswith('train'):
            typing_loss = torch.mean(torch.stack(proto_losses), dim=-1)

        if not stage.startswith('train'):
            self.__save_evaluate_predicted_result__(batch_result,
                                                    device_id=device_id,
                                                    stage=stage,
                                                    path=path)

        # return SpanProtoOutput(
        #         loss=((support_detector_outputs.loss + query_detector_outputs.loss) / 2.0 + typing_loss)
        #         if stage.startswith("train") else (support_detector_outputs.loss + query_detector_outputs.loss),
        #     ) # 返回部分的所有logits不论最外层是list还是tuple，最里层一定要包含一个张量，否则huggingface里的nested_detach函数会报错
        return SpanProtoOutput(
            loss=(support_detector_outputs.loss + typing_loss)
            if stage.startswith('train') else query_detector_outputs.loss,
        )  # 返回部分的所有logits不论最外层是list还是tuple，最里层一定要包含一个张量，否则huggingface里的nested_detach函数会报错

    def __save_evaluate_predicted_result__(self,
                                           new_result: dict,
                                           device_id: int = 0,
                                           stage='dev',
                                           path=None):
        '''
        本函数用于在forward时保存每一个batch内的预测span以及span type
        new_result / result: {
            '(id)': { # id-th episode query
                'spans': [[[1, 4], [6, 7], xxx], ... ] # [sent_num, span_num, 2]
                'types': [[2, 0, xxx], ...] # [sent_num, span_num]
            },
            xxx
        }
        '''
        # 拉取当前任务中已经预测的结果
        self.predict_dir = self.predict_result_path(path)
        npy_file_name = os.path.join(
            self.predict_dir, '{}_predictions_{}.npy'.format(stage, device_id))
        result = dict()
        if os.path.exists(npy_file_name):
            result = np.load(npy_file_name, allow_pickle=True)[()]
        # 合并
        for episode_id, query_res in new_result.items():
            result[episode_id] = query_res
        # 保存
        np.save(npy_file_name, result, allow_pickle=True)

    def get_topk_spans(self,
                       probs,
                       indices,
                       input_ids,
                       threshold=0.60,
                       low_threshold=0.1,
                       is_query=False):
        '''
        probs: [n, m]
        indices: [n, m]
        input_texts: [n, seq_len]
        is_query: if true, each sentence must recall at least one span
        '''
        probs = probs.squeeze(
            1).detach().cpu()  # topk结果的概率 [n, m]  # 返回的已经是按照概率进行降序排列的结果
        indices = indices.squeeze(
            1).detach().cpu()  # topk结果的索引 [n, m]  # 返回的已经是按照概率进行降序排列的结果
        input_ids = input_ids.detach().cpu()
        # print('probs=', probs) # [n, m]
        # print('indices=', indices) # [n, m]
        predict_span = list()
        if is_query:
            low_threshold = 0.0
        for prob, index, text in zip(probs, indices,
                                     input_ids):  # 遍历每个句子，其对应若干预测的span及其概率
            threshold_ = threshold
            index_ids = torch.Tensor([i for i in range(len(index))]).long()
            span = set()
            # TODO 1. 调节阈值 2. 处理输出实体重叠问题
            entity_index = index[prob >= low_threshold]
            index_ids = index_ids[prob >= low_threshold]
            while threshold_ >= low_threshold:  # 动态控制阈值，以确保可以召回出span数量是尽可能均匀的（如果所有句子使用同一个阈值，那么每个句子被召回的span数量参差不齐）
                for ei, entity in enumerate(entity_index):
                    p = prob[index_ids[ei]]
                    if p < threshold_:  # 如果此时候选的span得分已经低于阈值，由于获得的结果已经是降序排列的，则后续的结果一定都低于阈值，则直接结束
                        break
                    # 1D index转2D index
                    start_end = np.unravel_index(
                        entity, (self.max_length, self.max_length))
                    # print('self.max_length=', self.max_length)
                    s, e = start_end[0], start_end[1]
                    ans = text[s:e]
                    # if ans not in answer:
                    #     answer.append(ans)
                    #     topk_answer_dict[ans] = {'prob': float(prob[index_ids[ei]]), 'pos': [(s, e)]}
                    span.add((s, e))
                # 满足下列几个条件的，动态调低阈值，并重新筛选
                if len(span) <= 3:
                    threshold_ -= 0.05
                else:
                    break
            if len(span) == 0:
                # 如果当前没有召回出任何span，则直接选择[cls]作为结果（相当于MRC的unanswerable）
                span = [[0, 0]]
            span = [list(i) for i in list(span)]
            # print("prob=", prob) e.g. [0.96, 0.85, 0.04, 0.00, ...]
            # print("span=", span) e.g. [[20, 23], [11, 14]]
            predict_span.append(span)
        return predict_span

    def split_span(self,
                   labeled_spans: list,
                   labeled_types: list,
                   predict_spans: list,
                   stage: str = 'train'):
        """# 对detector预测的所有span，划分出哪些是labeled span，哪些是unlabeled span."""
        def check_similar_span(span1, span2):
            """检测两个span是否接近，例如[12, 16], [11, 16], [13, 15], [12, 17]是接近的."""
            # 考虑一个特殊情况，例如 [12, 12], [13, 13]
            if len(span1) == 0 or len(span2) == 0:
                return False
            if span1[0] == span1[1] and span2[0] == span2[1] and abs(
                    span1[0] - span2[0]) == 1:
                return False
            if abs(span1[0] - span2[0]) <= 1 and abs(
                    span1[1] - span2[1]) <= 1:  # 两个区间的起点和终点分别相差1以内
                return True
            return False

        all_spans, span_types = list(), list()  # [n, m]
        num = 0
        unlabeled_spans = list()
        for labeled_span, labeled_type, predict_span in zip(
                labeled_spans, labeled_types, predict_spans):
            # 对detector预测的所有span，划分出哪些是labeled span，哪些是unlabeled span
            unlabeled_span = list()
            # if len(all_span) != len(span_type):
            #     length = min(len(all_span), len(span_type))
            #     all_span, span_type = all_span[: length], span_type[: length]
            for span in predict_span:  # 遍历每个预测的span
                if span not in labeled_span:  # 如果span没有存在，则说明当前的span是unlabeled的
                    # 可能存在一些临界点非常接近的（global pointer预测的临界点有时候很模糊），对于临界点相近的予以排除
                    is_remove = False
                    for span_x in labeled_span:  # 遍历所有已经被merge的span
                        is_remove = check_similar_span(
                            span_x, span)  # 如果已存在的span，和当前的span很接近，则排除当前的span
                        if is_remove is True:
                            break
                    if is_remove is True:
                        continue
                    unlabeled_span.append(span)
            # if self.global_step % 1000 == 0:
            #     print(" === ")
            #     print('labeled_span=', labeled_span) # [[1, 3], [12, 14], [25, 25], [7, 7]]
            #     print('predict_span=', predict_span) # [[25, 25], [1, 3], [12, 14], [7, 7]]
            # if len(unlabeled_span) == 0 and stage.startswith("train"):
            #     # 如果当前句子没有一个unlabeled span，则需要进行负采样，以确保unlabeled不为空
            #     # print("unlabeled span is empty, so we randomly select one span as the unlabeled span")
            #     # all_span.append([0, 0])
            #     # span_type.append(self.num_class)
            #     while True:
            #         random_span = np.random.randint(0, 32, 2).tolist()
            #         if abs(random_span[0] - random_span[1]) > 10:
            #             continue
            #         random_span = [random_span[1], random_span[0]] if random_span[0] > random_span[1] else random_span
            #         if random_span in labeled_span or random_span in unlabeled_span:
            #             continue
            #         unlabeled_span.append(random_span)
            #         break
            num += len(unlabeled_span)
            unlabeled_spans.append(unlabeled_span)
        # print("num=", num)
        return unlabeled_spans

    def merge_span(self,
                   labeled_spans: list,
                   labeled_types: list,
                   predict_spans: list,
                   stage: str = 'train'):
        def check_similar_span(span1, span2):
            """检测两个span是否接近，例如[12, 16], [11, 16], [13, 15], [12, 17]是接近的."""
            # 考虑一个特殊情况，例如 [12, 12], [13, 13]
            if len(span1) == 0 or len(span2) == 0:
                return False
            if span1[0] == span1[1] and span2[0] == span2[1] and abs(
                    span1[0] - span2[0]) == 1:
                return False
            if abs(span1[0] - span2[0]) <= 1 and abs(
                    span1[1] - span2[1]) <= 1:  # 两个区间的起点和终点分别相差1以内
                return True
            return False

        all_spans, span_types = list(), list()  # [n, m]
        for labeled_span, labeled_type, predict_span in zip(
                labeled_spans, labeled_types, predict_spans):
            # 遍历每个句子，对它们的span进行合并
            unlabeled_num = 0
            all_span, span_type = labeled_span, labeled_type  # 先加入所有labeled span
            if len(all_span) != len(span_type):
                length = min(len(all_span), len(span_type))
                all_span, span_type = all_span[:length], span_type[:length]
            for span in predict_span:  # 遍历每个预测的span
                if span not in all_span:  # 如果span没有存在，则说明当前的span是unlabeled的
                    # 可能存在一些临界点非常接近的（global pointer预测的临界点有时候很模糊），对于临界点相近的予以排除
                    is_remove = False
                    for span_x in all_span:  # 遍历所有已经被merge的span
                        is_remove = check_similar_span(
                            span_x, span)  # 如果已存在的span，和当前的span很接近，则排除当前的span
                        if is_remove is True:
                            break
                    if is_remove is True:
                        continue
                    all_span.append(span)
                    span_type.append(
                        self.num_class
                    )  # e.g. 5-way问题，已有标签为0，1，2，3，4，因此新增一个标签为5
                    unlabeled_num += 1
            # if self.global_step % 1000 == 0:
            #     print(" === ")
            #     print('labeled_span=', labeled_span) # [[1, 3], [12, 14], [25, 25], [7, 7]]
            #     print('predict_span=', predict_span) # [[25, 25], [1, 3], [12, 14], [7, 7]]
            if unlabeled_num == 0 and stage.startswith('train'):
                # 如果当前句子没有一个unlabeled span，则需要进行负采样，以确保unlabeled不为空
                # print("unlabeled span is empty, so we randomly select one span as the unlabeled span")
                # all_span.append([0, 0])
                # span_type.append(self.num_class)
                while True:
                    random_span = np.random.randint(0, 32, 2).tolist()
                    if abs(random_span[0] - random_span[1]) > 10:
                        continue
                    random_span = [
                        random_span[1], random_span[0]
                    ] if random_span[0] > random_span[1] else random_span
                    if random_span in all_span:
                        continue
                    all_span.append(random_span)
                    span_type.append(self.num_class)
                    break

            # if len(all_span) != len(span_type):
            #     all_span = [[0, 0]]
            #     span_type = [self.num_class]

            all_spans.append(all_span)
            span_types.append(span_type)

        return all_spans, span_types
