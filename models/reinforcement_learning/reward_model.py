# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 4:29 p.m.
# @Author  : JianingWang
# @File    : reward_model.py

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from loss.rl_loss import LogSigLoss, LogExpLoss
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel, GPT2Model

"""
RoERTa for Reward Model
"""
class RobertaForReward(RobertaPreTrainedModel):
    """
    Reward model base class.

    Args:
        model (nn.Module): Reward model.
        value_head (nn.Module): Value head to get reward score.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.roberta = RobertaModel(config)
        self.value_head = nn.Linear(self.config.n_embd, 1)
        self.init_weights()

    def forward(
            self, 
            chosen_sequences: torch.LongTensor, 
            chosen_attention_mask: Optional[torch.Tensor],
            rejected_sequences: Optional[torch.LongTensor] = None,
            rejected_attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        # obtain reward value of chosen sequence
        chosen_outputs = self.roberta(chosen_sequences, attention_mask=chosen_attention_mask)
        chosen_last_hidden_states = chosen_outputs['last_hidden_state']
        chosen_values = self.value_head(chosen_last_hidden_states)[:, :-1]
        chosen_values = chosen_values.mean(dim=1).squeeze(1)    # ensure shape is (B)

        return_dict = {
            "chosen_values": chosen_values,
        }
        # if has rejected, obtain reward of rejected sequence, and calculate the loss
        if rejected_sequences is not None:
            rejected_outputs = self.roberta(rejected_sequences, attention_mask=rejected_attention_mask)
            rejected_last_hidden_states = rejected_outputs['last_hidden_state']
            rejected_values = self.value_head(rejected_last_hidden_states)[:, :-1]
            rejected_values = rejected_values.mean(dim=1).squeeze(1)    # ensure shape is (B)
            return_dict["rejected_values"] = rejected_values
            
            loss_fn = LogSigLoss()
            loss = loss_fn(chosen_values, rejected_values)
            
            return_dict["loss"] = loss
        
        return return_dict


"""
GPT2 for Reward Model
"""
class GPT2ForReward(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]
    """
    Reward model base class.

    Args:
        model (nn.Module): Reward model.
        value_head (nn.Module): Value head to get reward score.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.transformer = GPT2Model(config)
        self.value_head = nn.Linear(self.config.n_embd, 1)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        self.post_init()

    def forward(
            self, 
            chosen_sequences: torch.LongTensor, 
            chosen_attention_mask: Optional[torch.Tensor],
            rejected_sequences: Optional[torch.LongTensor] = None,
            rejected_attention_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        # obtain reward value of chosen sequence
        chosen_outputs = self.transformer(chosen_sequences, attention_mask=chosen_attention_mask)
        chosen_last_hidden_states = chosen_outputs['last_hidden_state']
        chosen_values = self.value_head(chosen_last_hidden_states)[:, :-1]
        chosen_values = chosen_values.mean(dim=1).squeeze(1)    # ensure shape is (B)

        return_dict = {
            "chosen_values": chosen_values,
        }
        # if has rejected, obtain reward of rejected sequence, and calculate the loss
        if rejected_sequences is not None:
            rejected_outputs = self.transformer(rejected_sequences, attention_mask=rejected_attention_mask)
            rejected_last_hidden_states = rejected_outputs['last_hidden_state']
            rejected_values = self.value_head(rejected_last_hidden_states)[:, :-1]
            rejected_values = rejected_values.mean(dim=1).squeeze(1)    # ensure shape is (B)
            return_dict["rejected_values"] = rejected_values
            loss_fn = LogSigLoss()
            loss = loss_fn(chosen_values, rejected_values)
            
            return_dict["loss"] = loss
        
        return return_dict
    
    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )