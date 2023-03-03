# -*- coding: utf-8 -*-
# @Time    : 2022/4/12 12:12 下午
# @Author  : JianingWang
# @File    : duma.py
import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from transformers.models.albert.modeling_albert import AlbertModel, AlbertPreTrainedModel
from transformers.models.megatron_bert.modeling_megatron_bert import MegatronBertModel, MegatronBertPreTrainedModel
from transformers.modeling_outputs import MultipleChoiceModelOutput


def split_context_query(sequence_output, pq_end_pos, input_ids):
    context_max_len = sequence_output.size(1)
    query_max_len = sequence_output.size(1)
    sep_tok_len = 1  # [SEP]
    context_sequence_output = sequence_output.new(
        torch.Size((sequence_output.size(0), context_max_len, sequence_output.size(2)))).zero_()
    query_sequence_output = sequence_output.new_zeros(
        (sequence_output.size(0), query_max_len, sequence_output.size(2)))
    query_attention_mask = sequence_output.new_zeros((sequence_output.size(0), query_max_len))
    context_attention_mask = sequence_output.new_zeros((sequence_output.size(0), context_max_len))
    for i in range(0, sequence_output.size(0)):
        p_end = pq_end_pos[i][0]
        q_end = pq_end_pos[i][1]
        context_sequence_output[i, :min(context_max_len, p_end)] = sequence_output[i, 1: 1 + min(context_max_len, p_end)]
        idx = min(query_max_len, q_end - p_end - sep_tok_len)
        query_sequence_output[i, :idx] = sequence_output[i, p_end + sep_tok_len + 1: p_end + sep_tok_len + 1 + min(q_end - p_end - sep_tok_len, query_max_len)]
        query_attention_mask[i, :idx] = sequence_output.new_ones((1, query_max_len))[0, :idx]
        context_attention_mask[i, : min(context_max_len, p_end)] = sequence_output.new_ones((1, context_max_len))[0, : min(context_max_len, p_end)]
    return context_sequence_output, query_sequence_output, context_attention_mask, query_attention_mask


class BertCoAttention(nn.Module):
    def __init__(self, config):
        super(BertCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, context_states, query_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        mixed_query_layer = self.query(query_states)

        extended_attention_mask = attention_mask[:, None, None, :]
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask = extended_attention_mask

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder"s padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(context_states)
            mixed_value_layer = self.value(context_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        outputs = context_layer
        return outputs


class BertDUMAForMultipleChoice(BertPreTrainedModel):

    def __init__(self, config):
        super(BertDUMAForMultipleChoice, self).__init__(config)

        self.bert = BertModel(config)
        self.classifier_2 = nn.Linear(2 * config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert_att = BertCoAttention(config)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, pq_end_pos=None, iter=1):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_head_mask = head_mask.view(-1, head_mask.size(-1)) if head_mask is not None else None
        flat_inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-1)) if inputs_embeds is not None else None

        outputs = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            head_mask=flat_head_mask,
            inputs_embeds=flat_inputs_embeds
        )

        sequence_output = outputs[0]

        pq_end_pos = pq_end_pos.view(-1, pq_end_pos.size(-1))

        context_sequence_output, query_sequence_output, context_attention_mask, query_attention_mask = \
            split_context_query(sequence_output, pq_end_pos, input_ids)
        for _ in range(0, iter):
            cq_biatt_output = self.bert_att(context_sequence_output, query_sequence_output, context_attention_mask)
            qc_biatt_output = self.bert_att(query_sequence_output, context_sequence_output, query_attention_mask)

            query_sequence_output = cq_biatt_output
            context_sequence_output = qc_biatt_output

        cat_output = torch.cat([torch.mean(qc_biatt_output, 1), torch.mean(cq_biatt_output, 1)], 1)
        pooled_output = self.dropout(cat_output)
        logits = self.classifier_2(pooled_output)

        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class RobertaDUMAForMultipleChoice(RobertaPreTrainedModel):

    def __init__(self, config):
        super(RobertaDUMAForMultipleChoice, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.classifier_2 = nn.Linear(2 * config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert_att = BertCoAttention(config)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, pq_end_pos=None, iter=1):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_head_mask = head_mask.view(-1, head_mask.size(-1)) if head_mask is not None else None
        flat_inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-1)) if inputs_embeds is not None else None

        outputs = self.roberta(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            head_mask=flat_head_mask,
            inputs_embeds=flat_inputs_embeds
        )

        sequence_output = outputs[0]

        pq_end_pos = pq_end_pos.view(-1, pq_end_pos.size(-1))

        context_sequence_output, query_sequence_output, context_attention_mask, query_attention_mask = \
            split_context_query(sequence_output, pq_end_pos, input_ids)
        for _ in range(0, iter):
            cq_biatt_output = self.bert_att(context_sequence_output, query_sequence_output, context_attention_mask)
            qc_biatt_output = self.bert_att(query_sequence_output, context_sequence_output, query_attention_mask)

            query_sequence_output = cq_biatt_output
            context_sequence_output = qc_biatt_output

        cat_output = torch.cat([torch.mean(qc_biatt_output, 1), torch.mean(cq_biatt_output, 1)], 1)
        pooled_output = self.dropout(cat_output)
        logits = self.classifier_2(pooled_output)

        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)

class AlbertDUMAForMultipleChoice(AlbertPreTrainedModel):

    def __init__(self, config):
        super(AlbertDUMAForMultipleChoice, self).__init__(config)

        self.albert = AlbertModel(config)
        self.classifier_2 = nn.Linear(2 * config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert_att = BertCoAttention(config)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, pq_end_pos=None, iter=1):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_head_mask = head_mask.view(-1, head_mask.size(-1)) if head_mask is not None else None
        flat_inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-1)) if inputs_embeds is not None else None

        outputs = self.albert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            head_mask=flat_head_mask,
            inputs_embeds=flat_inputs_embeds
        )

        sequence_output = outputs[0]

        pq_end_pos = pq_end_pos.view(-1, pq_end_pos.size(-1))

        context_sequence_output, query_sequence_output, context_attention_mask, query_attention_mask = \
            split_context_query(sequence_output, pq_end_pos, input_ids)
        for _ in range(0, iter):
            cq_biatt_output = self.bert_att(context_sequence_output, query_sequence_output, context_attention_mask)
            qc_biatt_output = self.bert_att(query_sequence_output, context_sequence_output, query_attention_mask)

            query_sequence_output = cq_biatt_output
            context_sequence_output = qc_biatt_output

        cat_output = torch.cat([torch.mean(qc_biatt_output, 1), torch.mean(cq_biatt_output, 1)], 1)
        pooled_output = self.dropout(cat_output)
        logits = self.classifier_2(pooled_output)

        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class MegatronDumaForMultipleChoice(MegatronBertPreTrainedModel):

    def __init__(self, config):
        super(MegatronDumaForMultipleChoice, self).__init__(config)

        self.bert = MegatronBertModel(config)
        self.classifier_2 = nn.Linear(2 * config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert_att = BertCoAttention(config)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, pq_end_pos=None, iter=1):
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_head_mask = head_mask.view(-1, head_mask.size(-1)) if head_mask is not None else None
        flat_inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-1)) if inputs_embeds is not None else None

        outputs = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            head_mask=flat_head_mask,
            inputs_embeds=flat_inputs_embeds
        )

        sequence_output = outputs[0]

        pq_end_pos = pq_end_pos.view(-1, pq_end_pos.size(-1))

        context_sequence_output, query_sequence_output, context_attention_mask, query_attention_mask = \
            split_context_query(sequence_output, pq_end_pos, input_ids)
        for _ in range(0, iter):
            cq_biatt_output = self.bert_att(context_sequence_output, query_sequence_output, context_attention_mask)
            qc_biatt_output = self.bert_att(query_sequence_output, context_sequence_output, query_attention_mask)

            query_sequence_output = cq_biatt_output
            context_sequence_output = qc_biatt_output

        cat_output = torch.cat([torch.mean(qc_biatt_output, 1), torch.mean(cq_biatt_output, 1)], 1)
        pooled_output = self.dropout(cat_output)
        logits = self.classifier_2(pooled_output)

        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)
