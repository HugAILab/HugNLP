# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 3:53 p.m.
# @Author  : JianingWang
# @File    : actor.py

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoConfig
from models.basic_modules.generation import generate


"""
Actor model.
"""
class Actor(nn.Module):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
    
    def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
        return log_probs_labels.squeeze(-1)

    """
    For generative model, needs generate function.
    """
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        return_action_mask: bool = True,
        **kwargs
    ) -> Union[Tuple[torch.LongTensor, torch.LongTensor], Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]]:
        sequences = generate(self.model, input_ids, **kwargs)
        attention_mask = None
        pad_token_id = kwargs.get('pad_token_id', None)
        if pad_token_id is not None:
            attention_mask = sequences.not_equal(pad_token_id).to(dtype=torch.long, device=sequences.device)
        if not return_action_mask:
            return sequences, attention_mask, None
        input_len = input_ids.size(1)
        eos_token_id = kwargs.get('eos_token_id', None)
        if eos_token_id is None:
            action_mask = torch.ones_like(sequences, dtype=torch.bool)
        else:
            # left padding may be applied, only mask action
            action_mask = (sequences[:, input_len:] == eos_token_id).cumsum(dim=-1) == 0
            action_mask = F.pad(action_mask, (1 + input_len, -1), value=True)    # include eos token and input
        action_mask[:, :input_len] = False
        action_mask = action_mask[:, 1:]
        return sequences, attention_mask, action_mask[:, -(sequences.size(1) - input_len):]

    def forward(self,
                sequences: torch.LongTensor,
                num_actions: int,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns action log probs
        """
        output = self.model(sequences, attention_mask=attention_mask)
        logits = output['logits']
        log_probs = self.log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
        return log_probs[:, -num_actions:]

    def get_base_model(self):
        return self.model


"""
Causal LM as a actor, e.g., GPT-2, OPT, BLOOM, etc.
"""
class CausalActor(Actor):
    """
    Causal LM Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (AutoConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
    """

    def __init__(self,
                 pretrained: str = None,
                 config: Optional[AutoConfig] = None,
                 checkpoint: bool = False) -> None:
        if pretrained is not None:
            model = AutoModelForCausalLM.from_pretrained(pretrained)
        elif config is not None:
            model = AutoModelForCausalLM(config)
        else:
            model = AutoModelForCausalLM(AutoConfig())
        if checkpoint:
            model.gradient_checkpointing_enable()
        super().__init__(model)