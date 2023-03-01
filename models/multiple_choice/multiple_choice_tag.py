# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 7:59 下午
# @Author  : JianingWang
# @File    : multiple_choice.py
import torch
from roformer import RoFormerPreTrainedModel, RoFormerModel
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import MegatronBertPreTrainedModel, MegatronBertModel
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.models.bert import BertPreTrainedModel, BertModel


class BertForTagMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None else
                              config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size * 2, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
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
                pseudo=None):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[
            1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(
            -1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(
            -1,
            attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(
            -1,
            token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(
            -1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (inputs_embeds.view(-1, inputs_embeds.size(-2),
                                            inputs_embeds.size(-1))
                         if inputs_embeds is not None else None)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        w = torch.logical_and(input_ids >= min(self.config.start_token_ids),
                              input_ids <= max(self.config.start_token_ids))
        start_index = w.nonzero()[:, 1].view(-1, 2)
        # <start_entity> + <end_entity> 进分类
        pooled_output = torch.cat([
            torch.cat([x[y[0], :], x[y[1], :]]).unsqueeze(0)
            for x, y in zip(outputs.last_hidden_state, start_index)
        ])

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            if pseudo is None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(reshaped_logits, labels)
            else:
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(reshaped_logits, labels)
                weight = 1 - pseudo * 0.9
                loss *= weight
                loss = loss.mean()

        if not return_dict:
            output = (reshaped_logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RoFormerForTagMultipleChoice(RoFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roformer = RoFormerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[
            1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(
            -1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(
            -1,
            attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(
            -1,
            token_type_ids.size(-1)) if token_type_ids is not None else None

        inputs_embeds = (inputs_embeds.view(-1, inputs_embeds.size(-2),
                                            inputs_embeds.size(-1))
                         if inputs_embeds is not None else None)

        outputs = self.roformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        w = torch.logical_and(input_ids >= min(self.config.start_token_ids),
                              input_ids <= max(self.config.start_token_ids))
        start_index = w.nonzero()[:, 1].view(-1, 2)
        # <start_entity> + <end_entity> 进分类
        pooled_output = torch.cat([
            torch.cat([x[y[0], :], x[y[1], :]]).unsqueeze(0)
            for x, y in zip(outputs.last_hidden_state, start_index)
        ])

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MegatronBertForTagMultipleChoice(MegatronBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = MegatronBertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.hidden_size * 2, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self,
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
                pseudo=None):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[
            1] if input_ids is not None else inputs_embeds.shape[1]
        input_ids = input_ids.view(
            -1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(
            -1,
            attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(
            -1,
            token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(
            -1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (inputs_embeds.view(-1, inputs_embeds.size(-2),
                                            inputs_embeds.size(-1))
                         if inputs_embeds is not None else None)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        w = torch.logical_and(input_ids >= min(self.config.start_token_ids),
                              input_ids <= max(self.config.start_token_ids))
        start_index = w.nonzero()[:, 1].view(-1, 2)
        # <start_entity> + <end_entity> 进分类
        pooled_output = torch.cat([
            torch.cat([x[y[0], :], x[y[1], :]]).unsqueeze(0)
            for x, y in zip(outputs.last_hidden_state, start_index)
        ])

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            if pseudo is None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(reshaped_logits, labels)
            else:
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(reshaped_logits, labels)
                weight = 1 - pseudo * 0.9
                loss *= weight
                loss = loss.mean()

        if not return_dict:
            output = (reshaped_logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
