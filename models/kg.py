# -*- coding: utf-8 -*-
# @Time    : 2022/2/17 11:26 上午
# @Author  : JianingWang
# @File    : kg.py
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from collections import OrderedDict
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class BertForPretrainWithKG(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.cls = BertOnlyMLMHead(config)
        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, config.num_ner_labels) for _ in range(config.entity_type_num)])
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            ner_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state
        # mlm
        prediction_scores = self.cls(sequence_output)
        # ner
        sequence_output = self.dropout(sequence_output)
        ner_logits = torch.stack([classifier(sequence_output) for classifier in self.classifiers]).movedim(1, 0)

        # mlm
        masked_lm_loss, ner_loss, total_loss = None, None, None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if ner_labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss

            active_loss = attention_mask.repeat(self.config.entity_type_num, 1, 1).view(-1) == 1
            active_logits = ner_logits.reshape(-1, self.config.num_ner_labels)
            active_labels = torch.where(
                active_loss, ner_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(ner_labels)
            )
            ner_loss = loss_fct(active_logits, active_labels)

        if masked_lm_loss:
            total_loss = masked_lm_loss + ner_loss * 4

        return OrderedDict([
            ('loss', total_loss),
            ('mlm_loss', masked_lm_loss.unsqueeze(0)),
            ('ner_loss', ner_loss.unsqueeze(0)),
            ('logits', prediction_scores.argmax(2)),
            ('ner_logits', ner_logits.argmax(3))
        ])
        # MaskedLMOutput(
        #     loss=total_loss,
        #     logits=prediction_scores.argmax(2),
        #     ner_l
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )


class BertForPretrainWithKGV2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.cls = BertOnlyMLMHead(config)
        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, config.num_ner_labels) for _ in range(config.entity_type_num)])
        self.mlp = MLPLayer(config)
        self.sim = Similarity(0.05)
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            ner_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state
        # mlm
        prediction_scores = self.cls(sequence_output)
        # ner
        sequence_output = self.dropout(sequence_output)
        ner_logits = torch.stack([classifier(sequence_output) for classifier in self.classifiers]).movedim(1, 0)

        # mlm
        masked_lm_loss, ner_loss, total_loss = None, None, None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if ner_labels is not None:
            loss_fct = CrossEntropyLoss()
            active_logits = ner_logits.reshape(-1, self.config.num_ner_labels)
            # padding 的label是-100
            ner_loss = loss_fct(active_logits, ner_labels.view(-1))

        if masked_lm_loss:
            total_loss = masked_lm_loss

        if ner_loss:
            total_loss = total_loss + ner_loss

        # 对比cls loss
        # cls_hidden = outputs.pooler_output
        cls_hidden = sequence_output[:, 0]
        simcse_loss = self.simcse_unsup_loss2(cls_hidden)
        if simcse_loss:
            total_loss = total_loss + simcse_loss*10

        ner_out = ner_logits.argmax(3)
        return OrderedDict([
            ('loss', total_loss),
            ('mlm_loss', masked_lm_loss.unsqueeze(0)),
            ('ner_loss', ner_loss.unsqueeze(0)),
            ('logits', prediction_scores.argmax(2)),
            ('ner_logits', ner_out.view(ner_out.shape[0], -1)),
            ('simcse_loss', simcse_loss.unsqueeze(0))
        ])

    def simcse_unsup_loss2(self, pooler_output):
        pooler_output = pooler_output.view((-1, 2, pooler_output.size(-1)))
        pooler_output = self.mlp(pooler_output)
        z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        labels = torch.arange(cos_sim.size(0)).long().to(pooler_output.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)
        return loss

    @staticmethod
    def simcse_unsup_loss(y_pred: 'tensor') -> 'tensor':
        # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
        y_true = torch.arange(y_pred.shape[0], device=y_pred.device)
        y_true = (y_true - y_true % 2 * 2) + 1
        # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        # sim = torch.mm(y_pred, y_pred.transpose(0, 1))
        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        sim = sim - torch.eye(y_pred.shape[0], device=y_pred.device) * 1e12
        # 相似度矩阵除以温度系数
        sim = sim/0.05
        # 计算相似度矩阵与y_true的交叉熵损失
        loss = F.cross_entropy(sim, y_true)
        print(loss)
        return loss
