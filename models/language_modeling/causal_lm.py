# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 3:35 下午
# @Author  : JianingWang
# @File    : mlm.py
import logging
from typing import Union, Tuple, Optional
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Tuple
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model, GPT2PreTrainedModel
"""
Function: Use Causal LM to pre-train GPT-2
Notes:
- In default, the Causal LM aims to train on all tokens, the label of each token is the next token, which let the model learn in regressive way.
- If you want to choose some tokens, or mask some tokens (like MLM), the label of non-masked token should be -100, which can be used for cross-entropy function (only calculate loss at not -100)
"""


class GPT2ForCausalLM(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r'attn.masked_bias', r'attn.bias', r'lm_head.weight'
    ]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get('token_type_ids', None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            'input_ids': input_ids,
            'past_key_values': past,
            'use_cache': kwargs.get('use_cache'),
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # print("shift_labels=", shift_labels)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits, ) + transformer_outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]],
                       beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or.

        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past) for layer_past in past)


# class GPT2ForCanusalLM(GPT2LMHeadModel):

#     def __init__(self, config):
#         super().__init__(config)
#         self.transformer = GPT2Model(config)
#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None, # input token id
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         label_masks: Optional[torch.LongTensor] = None, # mask=1 means it should be calculated loss
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         hidden_states = transformer_outputs[0]
#         lm_logits = self.lm_head(hidden_states)

#         # print("len(input_ids)=", len(input_ids[0]))
#         # print("input_ids[-1]=", input_ids[0][-1])

#         loss = None
#         if labels is not None:
#             shift_logits = lm_logits[..., :-1, :].contiguous()
#             # print("shift_logits.shape=", shift_logits.shape)
#             if labels is None:
#                 labels = input_ids
#             shift_labels = labels[..., 1:].contiguous()
#             # print("shift_labels=", shift_labels)
#             # print("shift_labels.shape=", shift_labels.shape)
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss(reduction="none")
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) # [batch_size, lngth]
#             label_masks = label_masks[..., 1:].contiguous()
#             # print("loss.shape=", loss.shape)
#             # print("shift_logits.shape=", shift_logits.shape)
#             # print("label_masks.shape=", label_masks.shape)
#             loss = loss.view(shift_logits.size(0), shift_logits.size(1)) * label_masks # [batch_size, length]
#             loss = torch.sum(loss, axis=1) / torch.sum(label_masks, axis=1) # [batch_size]
#             # print("loss=", loss)
#         if not return_dict:
#             output = (lm_logits,) + transformer_outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return CausalLMOutputWithCrossAttentions(
#             loss=loss,
#             logits=lm_logits,
#             past_key_values=transformer_outputs.past_key_values,
#             hidden_states=transformer_outputs.hidden_states,
#             attentions=transformer_outputs.attentions,
#             cross_attentions=transformer_outputs.cross_attentions,
#         )

if __name__ == '__main__':
    from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(
        '/Users/wangjianing/Desktop/开源代码与数据模型/模型/gpt2')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # print("tokenizer.eos_token_id=", tokenizer.eos_token_id) # 50256
    model = GPT2ForCanusalLM.from_pretrained(
        '/Users/wangjianing/Desktop/开源代码与数据模型/模型/gpt2')
    input_text = "My friend Jack invites me to play computer games with him, but my girl friend doesn't agree. I think"
    inputs = tokenizer(input_text,
                       add_special_tokens=True,
                       return_tensors='pt')
    inputs['labels'] = inputs['input_ids']
    print('inputs=', inputs)
    """
    inputs= {'input_ids': tensor([[ 3666,  1545,  3619, 27671,   502,   284,   711,  3644,  1830,   351,
           683,    11,   475,   616,  2576,  1545,  1595,   470,  4236,    13,
           314,   892,   220]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), 'labels': tensor([[ 3666,  1545,  3619, 27671,   502,   284,   711,  3644,  1830,   351,
           683,    11,   475,   616,  2576,  1545,  1595,   470,  4236,    13,
           314,   892,   220]])}

    """
    outputs = model(**inputs)
    print('loss=', outputs[0])
    """
    loss= tensor(3.9444, grad_fn=<NllLossBackward0>)
    """
    output_sequences = model.generate(
        **inputs,
        emb_match=None,
        control_code=None,
        past_key_values=None,
        max_length=len(inputs['input_ids'][0]) + 10,
        min_length=5,
        temperature=1.0,
        top_k=1,
        top_p=0.5,  #top_p=0.5,
        repetition_penalty=1.0,  # 重复词惩罚，用于控制生成多样性的文本
        do_sample=False,
        num_beams=5,
        # bad_words_ids=[[628], [198]] if True else None,
        num_return_sequences=1,
    )
    # print("output_sequences=", output_sequences)
    results = tokenizer.decode(output_sequences[0])
    print('results=', results)
    """
    results= My friend Jack invites me to play computer games with him, but my girl friend doesn't agree. I think  it's a good idea to play computer games
    """
