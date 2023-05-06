# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 4:29 p.m.
# @Author  : JianingWang
# @File    : reward_model.py

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


"""
Reward Model
"""
class RewardModel(nn.Module):
    """
    Reward model base class.

    Args:
        model (nn.Module): Reward model.
        value_head (nn.Module): Value head to get reward score.
    """

    def __init__(self,
                 model: nn.Module,
                 value_head: Optional[nn.Module] = None) -> None:
        self.model = model

        if value_head is not None:
            if value_head.out_features != 1:
                raise ValueError("The value head of reward model's output dim should be 1!")
            self.value_head = value_head
        else:
            self.value_head = nn.Linear(model.config.n_embd, 1)

    def forward(self, sequences: torch.LongTensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']
        values = self.value_head(last_hidden_states)[:, :-1]
        value = values.mean(dim=1).squeeze(1)    # ensure shape is (B)
        return value


"""
AutoModle for Reward
"""
class AutoModelReward(RewardModel):
    """
    AutoModel LM Reward model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (AutoConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.

    """

    def __init__(self,
                 pretrained: str = None,
                 config: Optional[AutoConfig] = None,
                 checkpoint: bool = False,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:
        if pretrained is not None:
            model = AutoModel.from_pretrained(pretrained)
        elif config is not None:
            model = AutoModel(config)
        else:
            model = AutoModel(AutoConfig())
        if checkpoint:
            model.gradient_checkpointing_enable()
        value_head = nn.Linear(model.config.hidden_size, 1)
        value_head.weight.data.normal_(mean=0.0, std=1 / (model.config.hidden_size + 1))
        super().__init__(model, value_head, lora_rank, lora_train_bias)