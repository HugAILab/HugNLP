# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 4:12 p.m.
# @Author  : JianingWang
# @File    : critic.py

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig
from models.basic_modules.generation import generate


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    return mean


"""
Critic model.
"""
class Critic(nn.Module):
    """
    Critic model base class.

    Args:
        model (nn.Module): Critic model.
        value_head (nn.Module): Value head to get value.
    """

    def __init__(
        self,
        model: nn.Module,
        value_head: nn.Module,
        use_action_mask: bool = False,
    ) -> None:

        self.model = model
        self.value_head = value_head # critic layer for predict value function
        self.use_action_mask = use_action_mask

    def forward(self,
                sequences: torch.LongTensor,
                action_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']

        values = self.value_head(last_hidden_states).squeeze(-1)

        if action_mask is not None and self.use_action_mask:
            num_actions = action_mask.size(1)
            prompt_mask = attention_mask[:, :-num_actions]
            values = values[:, :-num_actions]
            value = masked_mean(values, prompt_mask, dim=1)
            return value

        values = values[:, :-1]
        value = values.mean(dim=1)
        return value


"""
Auto Model for Critic
"""
class AutoModelCritic(Critic):
    """
    AutoModel Critic model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (AutoConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 config: Optional[AutoConfig] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none',
                 **kwargs) -> None:
        if pretrained is not None:
            model = AutoModel.from_pretrained(pretrained)
        elif config is not None:
            model = AutoModel(config)
        else:
            model = AutoModel(AutoConfig())
        if checkpoint:
            model.gradient_checkpointing_enable()
        value_head = nn.Linear(model.config.word_embed_proj_dim, 1)
        super().__init__(model, value_head, **kwargs)