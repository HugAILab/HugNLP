# -*- coding: utf-8 -*-
# @Time    : 2022/4/21 5:30 下午
# @Author  : JianingWang
# @File    : fusion_siamese.py
from typing import Optional
import torch
import numpy as np
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import BCEWithLogitsLoss
from transformers import MegatronBertModel, MegatronBertPreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.activations import ACT2FN
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from loss.focal_loss import FocalLoss
# from roformer import RoFormerPreTrainedModel, RoFormerModel


class BertPooler(nn.Module):
    def __init__(self, hidden_size, hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        # self.activation = nn.Tanh()
        self.activation = ACT2FN[hidden_act]
        # self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        return x


class BertForFusionSiamese(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act
        self.bert_poor = BertPooler(self.hidden_size, self.hidden_act)
        self.dense_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense_2 = nn.Linear(self.hidden_size, self.hidden_size)

        if hasattr(config, "cls_dropout_rate"):
            cls_dropout_rate = config.cls_dropout_rate
        else:
            cls_dropout_rate = config.hidden_dropout_prob
        self.dropout = nn.Dropout(cls_dropout_rate)
        self.classifier = nn.Linear(3 * self.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            pseudo_label=None,
            segment_spans=None,
            pseuso_proba=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        logits, outputs = None, None
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                  "position_ids": position_ids,
                  "head_mask": head_mask, "inputs_embeds": inputs_embeds, "output_attentions": output_attentions,
                  "output_hidden_states": output_hidden_states, "return_dict": return_dict}
        inputs = {k: v for k, v in inputs.items() if v is not None}
        outputs = self.bert(**inputs)
        if "sequence_output" in outputs:
            sequence_output = outputs.sequence_output # [bz, seq_len, dim]
        else:
            sequence_output = outputs[0] # [bz, seq_len, dim]

        cls_output = self.bert_poor(sequence_output) # [bz, dim]

        if segment_spans is not None:
            # 如果输入的是两个segment，则分别进行平均池化
            seg1_embeddings, seg2_embeddings = list(), list()
            for ei, sentence_embeddings in enumerate(sequence_output):
                # sentence_embedding: [seq_len, dim]
                seg1_start, seg1_end, seg2_start, seg2_end = segment_spans[ei]
                # print("sentence_embeddings[seg1_start, seg1_end].shape=", sentence_embeddings[seg1_start, seg1_end].shape)
                # print("torch.mean(sentence_embeddings[seg1_start, seg1_end], 0).shape=", torch.mean(sentence_embeddings[seg1_start, seg1_end], 0).shape)
                seg1_embeddings.append(torch.mean(sentence_embeddings[seg1_start: seg1_end], 0)) # [dim]
                seg2_embeddings.append(torch.mean(sentence_embeddings[seg2_start: seg2_end], 0)) # [dim]
            seg1_embeddings, seg2_embeddings = torch.stack(seg1_embeddings), torch.stack(seg2_embeddings) # [bz, dim]
            # print("seg1_embeddings.shape=", seg1_embeddings.shape)
            seg1_embeddings = self.bert_poor.activation(self.dense_1(seg1_embeddings))
            seg2_embeddings = self.bert_poor.activation(self.dense_1(seg2_embeddings))
            cls_output = torch.cat([cls_output, seg1_embeddings, seg2_embeddings], dim=-1) # [bz, 3*dim]
            # cls_output = cls_output + seg1_embeddings + seg2_embeddings # [bz, dim]

        pooler_output = self.dropout(cls_output)
        # pooler_output = self.LayerNorm(pooler_output)
        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:

            # loss_fct = FocalLoss()
            loss_fct = CrossEntropyLoss()
            # 伪标签
            if pseudo_label is not None:
                train_logits, pseudo_logits = logits[pseudo_label > 0.9], logits[pseudo_label < 0.1]
                train_labels, pseudo_labels = labels[pseudo_label > 0.9], labels[pseudo_label < 0.1]
                train_loss = loss_fct(train_logits.view(-1, self.num_labels),
                                      train_labels.view(-1)) if train_labels.nelement() else 0
                pseudo_loss = loss_fct(pseudo_logits.view(-1, self.num_labels),
                                       pseudo_labels.view(-1)) if pseudo_labels.nelement() else 0
                loss = 0.9 * train_loss + 0.1 * pseudo_loss
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class BertForWSC(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act
        self.bert_poor = BertPooler(self.hidden_size, self.hidden_act)
        self.dense_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense_2 = nn.Linear(self.hidden_size, self.hidden_size)

        if hasattr(config, "cls_dropout_rate"):
            cls_dropout_rate = config.cls_dropout_rate
        else:
            cls_dropout_rate = config.hidden_dropout_prob
        self.dropout = nn.Dropout(cls_dropout_rate)
        self.classifier = nn.Linear(2 * self.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            pseudo_label=None,
            span=None,
            pseuso_proba=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        logits, outputs = None, None
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                  "position_ids": position_ids,
                  "head_mask": head_mask, "inputs_embeds": inputs_embeds, "output_attentions": output_attentions,
                  "output_hidden_states": output_hidden_states, "return_dict": return_dict}
        inputs = {k: v for k, v in inputs.items() if v is not None}
        outputs = self.bert(**inputs)
        if "sequence_output" in outputs:
            sequence_output = outputs.sequence_output # [bz, seq_len, dim]
        else:
            sequence_output = outputs[0] # [bz, seq_len, dim]

        # cls_output = self.bert_poor(sequence_output) # [bz, dim]

        # 如果输入的是两个span，则分别进行平均池化
        seg1_embeddings, seg2_embeddings = list(), list()
        # print("span=", span)
        for ei, sentence_embeddings in enumerate(sequence_output):
            # sentence_embedding: [seq_len, dim]
            seg1_start, seg1_end, seg2_start, seg2_end = span[ei]
            # print("sentence_embeddings[seg1_start, seg1_end].shape=", sentence_embeddings[seg1_start, seg1_end].shape)
            # print("torch.mean(sentence_embeddings[seg1_start, seg1_end], 0).shape=", torch.mean(sentence_embeddings[seg1_start, seg1_end], 0).shape)
            seg1_embeddings.append(torch.mean(sentence_embeddings[seg1_start+1: seg1_end], 0)) # [dim]
            seg2_embeddings.append(torch.mean(sentence_embeddings[seg2_start+1: seg2_end], 0)) # [dim]
        seg1_embeddings, seg2_embeddings = torch.stack(seg1_embeddings), torch.stack(seg2_embeddings) # [bz, dim]
        # print("seg1_embeddings.shape=", seg1_embeddings.shape)
        # seg1_embeddings = self.bert_poor.activation(self.dense_1(seg1_embeddings))
        # seg2_embeddings = self.bert_poor.activation(self.dense_1(seg2_embeddings))
        cls_output = torch.cat([seg1_embeddings, seg2_embeddings], dim=-1) # [bz, 3*dim]
        # cls_output = cls_output + seg1_embeddings + seg2_embeddings # [bz, dim]

        pooler_output = self.dropout(cls_output)
        # pooler_output = self.LayerNorm(pooler_output)
        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:

            # loss_fct = FocalLoss()
            loss_fct = CrossEntropyLoss()
            # 伪标签
            if pseudo_label is not None:
                train_logits, pseudo_logits = logits[pseudo_label > 0.9], logits[pseudo_label < 0.1]
                train_labels, pseudo_labels = labels[pseudo_label > 0.9], labels[pseudo_label < 0.1]
                train_loss = loss_fct(train_logits.view(-1, self.num_labels),
                                      train_labels.view(-1)) if train_labels.nelement() else 0
                pseudo_loss = loss_fct(pseudo_logits.view(-1, self.num_labels),
                                       pseudo_labels.view(-1)) if pseudo_labels.nelement() else 0
                loss = 0.9 * train_loss + 0.1 * pseudo_loss
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
