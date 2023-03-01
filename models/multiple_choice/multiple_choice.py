# -*- coding: utf-8 -*-
# @Time    : 2022/4/16 12:10 下午
# @Author  : JianingWang
# @File    : multiple_choice.py
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
# from transformers import MegatronBertPreTrainedModel, MegatronBertModel
from transformers.models.megatron_bert import MegatronBertPreTrainedModel, MegatronBertModel
from transformers.modeling_outputs import MultipleChoiceModelOutput


class MegatronBertForMultipleChoice(MegatronBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = MegatronBertModel(config)
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        classifier_dropout = 0.2
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

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

        pooled_output = outputs[1]  # [batch_size, num_choices, hidden_dim]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_choices, 1]
        reshaped_logits = logits.view(-1,
                                      num_choices)  # [batch_size, num_choices]

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


class MegatronBertRDropForMultipleChoice(MegatronBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = MegatronBertModel(config)
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        classifier_dropout = 0.2
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

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
    ):
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

        logits_list = []
        for i in range(2):
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
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            logits_list.append(logits.view(-1, num_choices))

        loss = None
        alpha = 1.0
        for logits in logits_list:
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                l = loss_fct(logits, labels)
                if loss:
                    loss += alpha * l
                else:
                    loss = alpha * l

        if loss is not None:
            p = torch.log_softmax(logits_list[0], dim=-1)
            p_tec = torch.exp(p)
            q = torch.log_softmax(logits_list[-1], dim=-1)
            q_tec = torch.exp(q)

            kl_loss = F.kl_div(p, q_tec, reduction='none').sum()
            reverse_kl_loss = F.kl_div(q, p_tec, reduction='none').sum()
            loss += 0.5 * (kl_loss + reverse_kl_loss) / 2.

        return MultipleChoiceModelOutput(loss=loss,
                                         logits=logits_list[0],
                                         hidden_states=None,
                                         attentions=None)
