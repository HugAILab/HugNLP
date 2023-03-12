import sys
import os
import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import ModelOutput
from tools.runner_utils.log_util import logging
from tools.model_utils.parameter_freeze import ParameterFreeze

logger = logging.getLogger(__name__)
freezer = ParameterFreeze()


class PromptGPT2ModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    output_ids: Optional[torch.LongTensor] = None


"""
Function: Use Causal LM to prompt for cls
Notes:
- For classification, the model only calculate the loss at the position of label, the other position is set as -100
- During inference, generate result at the last position.
"""
class PromptGPT2ForSequenceClassification(torch.nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.gpt2_model = GPT2LMHeadModel(config=config)
        self.label_word_list = torch.Tensor(self.config.label_word_list).long().to(self.transformer.device)
        self.max_generation_len = self.label_word_list.shape[-1]
        self.num_log_probs = 1

    def from_pretrained(self, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        self.gpt2_model.from_pretrained(pretrained_model_name_or_path)


    # def get_label_probs(self, params, raw_resp):
    #     """Obtain model's label probability for one examples. The returned prob is NOT normalized

    #     raw_resp:
    #     {
    #         'text': ' I', # 当前生成的top词
    #         'logprobs': {'top_logprobs': [{' I': -3.4267239570617676, '\n': -3.5073862075805664, ...], # top100词及其socre
    #         'token_logprobs': [-3.4267239570617676], # 当前top词的score
    #         'tokens': [' I']}
    #     }
    #     """

    #     num_classes = self.label_word_list.shape[0]
    #     # approx = params['approx']
    #     # assert len(raw_resp) == len(test_sentences)

    #     # Fill in the labels that is in the top k prob
    #     top_logprobs = raw_resp['logprobs']['top_logprobs'][0]  # [0] since we only ask for complete one more token, dict
    #     # top_logprobs = {' I': -3.4267239570617676, '\n': -3.5073862075805664, ...}
    #     label_probs = [0] * num_classes
    #     for j, label_list in params['label_dict'].items():
    #         all_found = True
    #         for label in label_list:  # each possible label correspond to the same class
    #             label = " " + label  # notice prompt does not have space after 'A:'
    #             if label in top_logprobs:
    #                 label_probs[j] += np.exp(top_logprobs[label])
    #             else:
    #                 all_found = False

    #     return label_probs # NOT NORMALIZED


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

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.eval()

        assert input_ids.shape[0] == 1, "Generation model only need one example, not a batch size > 1"

        total_sequences = self.gpt2_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_generation_len + len(input_ids[0]),
            do_sample=False)


        # we are left padding, so we need to adjust the position IDs
        attention_mask = (total_sequences != 50256).float()
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # get the logits for the context and the next l tokens
        output = self.gpt2_model(
            input_ids=total_sequences,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True
        )
        logits = output.logits.detach().cpu()

        return PromptGPT2ModelOutput(
            loss=None,
            logits=logits,
            output_ids=total_sequences,
        )



# class PromptGPT2ForSequenceClassification(GPT2LMHeadModel):
#     _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

#     def __init__(self, config):
#         super().__init__(config)
#         self.transformer = GPT2Model(config)
#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

#         # if self.config.use_freezing:
#         #     self.transformer = freezer.freeze_lm(self.transformer)

#         # Model parallel
#         self.model_parallel = False
#         self.device_map = None

#         # Initialize weights and apply final processing
#         self.post_init()

#         # These attributes should be assigned once the model is initialized
#         self.label_word_list = torch.Tensor(self.config.label_word_list).long().to(self.transformer.device)
#         self.max_generation_len = self.label_word_list.shape[-1]

#     def get_output_embeddings(self):
#         return self.lm_head

#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings

#     def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
#         token_type_ids = kwargs.get("token_type_ids", None)
#         # only last token for inputs_ids if past is defined in kwargs
#         if past:
#             input_ids = input_ids[:, -1].unsqueeze(-1)
#             if token_type_ids is not None:
#                 token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

#         attention_mask = kwargs.get("attention_mask", None)
#         position_ids = kwargs.get("position_ids", None)

#         if attention_mask is not None and position_ids is None:
#             # create position_ids on the fly for batch generation
#             position_ids = attention_mask.long().cumsum(-1) - 1
#             position_ids.masked_fill_(attention_mask == 0, 1)
#             if past:
#                 position_ids = position_ids[:, -1].unsqueeze(-1)
#         else:
#             position_ids = None
#         return {
#             "input_ids": input_ids,
#             "past_key_values": past,
#             "use_cache": kwargs.get("use_cache"),
#             "position_ids": position_ids,
#             "attention_mask": attention_mask,
#             "token_type_ids": token_type_ids,
#         }

#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
#             `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
#             are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         transformer_outputs = self.transformer(
#             input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_attention_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         hidden_states = transformer_outputs[0]

#         # Set device for model parallelism
#         if self.model_parallel:
#             torch.cuda.set_device(self.transformer.first_device)
#             hidden_states = hidden_states.to(self.lm_head.weight.device)

#         lm_logits = self.lm_head(hidden_states)

#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = lm_logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # print("shift_labels=", shift_labels)
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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


#     @staticmethod
#     def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
#         """
#         This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
#         [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
#         beam_idx at every generation step.
#         """
#         return tuple(
#             tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
#             for layer_past in past
#         )




# if __name__ == "__main__":
#     from transformers import GPT2Tokenizer
#     tokenizer = GPT2Tokenizer.from_pretrained("/Users/wangjianing/Desktop/开源代码与数据模型/模型/gpt2")
#     model = GPT2ForInContextLearning.from_pretrained("/Users/wangjianing/Desktop/开源代码与数据模型/模型/gpt2")

#     # In-Context Learning for classification
#     # input_text = "The capital city of China is Beijing. \n\n The capital city of Japan is Tokyo. \n\n The capital city of America is"
#     input_text = "What are follows emotions? \n\n Input: The book is very nice.\n Output: Great. \n\n Input: I never eat chocolate!\n Output:"
#     # input_text = "This film is wonderful.\n Great."
#     tokenizer.pad_token = tokenizer.eos_token
#     inputs = tokenizer(input_text, return_tensors="pt")
#     input_len = inputs["input_ids"].shape[-1]
#     gen_output = model.generate(**inputs, max_length=input_len + 10)
#     gen_result = tokenizer.decode(gen_output[0])
#     print("classification result:\n", gen_result)

#     # In-Context Learning for generation
#     input_text = "Please tell me what is the transformer? "
#     # input_text = "This film is wonderful.\n Great."
#     tokenizer.pad_token = tokenizer.eos_token
#     inputs = tokenizer(input_text, return_tensors="pt")
#     input_len = inputs["input_ids"].shape[-1]
#     gen_output = model.generate(**inputs, max_length=input_len + 60)
#     gen_result = tokenizer.decode(gen_output[0])
#     print("generation result:\n", gen_result)
