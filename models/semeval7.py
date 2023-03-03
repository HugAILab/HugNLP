# -*- coding: utf-8 -*-
# @Time    : 2022/1/28 5:38 下午
# @Author  : JianingWang
# @File    : semeval7.py
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, DebertaV2Model, DebertaV2PreTrainedModel, StableDropout


class DebertaV2ForSemEval7MultiTask(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.num_labels = 3
        self.dense = nn.Linear(config.pooler_hidden_size*2, config.pooler_hidden_size)
        self.classifier = nn.Linear(output_dim, self.num_labels)
        self.regression = nn.Linear(output_dim, 1)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.dropout = StableDropout(drop_out)
        self.post_init()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            score=None
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        w = torch.logical_and(input_ids >= min(self.config.start_token_ids), input_ids <= max(self.config.start_token_ids))
        start_index = w.nonzero()[:, 1].view(-1, 2)
        # <start_entity> + <end_entity> 进分类
        pooler_output = torch.cat([torch.cat([x[y[0], :], x[y[1], :]]).unsqueeze(0) for x, y in zip(outputs.last_hidden_state, start_index)])
        # [CLS] + <start_entity> + <end_entity> 进分类
        # pooler_output = torch.cat([torch.cat([z, x[y[0], :], x[y[1], :]]).unsqueeze(0)
        #                            for x, y, z in zip(outputs.last_hidden_state, start_index, outputs.last_hidden_state[:, 0])])

        context_token = self.dropout(pooler_output)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        pooled_output = self.dropout(pooled_output)
        re_logits = self.regression(pooled_output)
        cls_logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            re_loss_func = MSELoss()
            re_loss = re_loss_func(re_logits.squeeze(), score.squeeze())

            cls_loss_func = CrossEntropyLoss()
            cls_loss = cls_loss_func(cls_logits.view(-1, self.num_labels), labels.view(-1))

            loss = re_loss + cls_loss

        return SequenceClassifierOutput(
            loss=loss, logits=torch.cat((cls_logits, re_logits), 1), hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
