# -*- coding: utf-8 -*-
# @Time    : 2021/11/28 5:25 下午
# @Author  : JianingWang
# @File    : data_collator.py
import random
import torch
import warnings
from itertools import chain
from typing import Any, Optional, Tuple, List, Union, Dict
from dataclasses import dataclass
from transformers import BatchEncoding, BertTokenizer, BertTokenizerFast, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling, _torch_collate_batch, tolist, DataCollatorMixin
from transformers.file_utils import PaddingStrategy
import time
"""
Standard data collator from HuggingFace.
"""


@dataclass
class DataCollatorForMaskedLM(DataCollatorForLanguageModeling):
    def torch_mask_tokens(
            self,
            inputs: Any,
            special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask,
                                               dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer),
                                     labels.shape,
                                     dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


"""
Data collator for token masking without number
"""


@dataclass
class DataCollatorForMaskedLMWithoutNumber(DataCollatorForLanguageModeling):
    """Mask token without number."""
    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                'This tokenizer does not have a mask token which is necessary for masked language modeling. '
                'You should pass `mlm=False` to train on causal language modeling instead.'
            )
        self.numerical_tokens = [
            v for k, v in self.tokenizer.vocab.items() if k.isdigit()
        ]
        self.exclude_tokens = set(self.numerical_tokens +
                                  self.tokenizer.all_special_ids)

    def torch_mask_tokens(
            self,
            inputs: Any,
            special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # special_tokens_mask2 = [
        #     [1 if token in self.exclude_tokens else 0 for token in val] for val in labels.tolist()
        # ]
        # special_tokens_mask2 = torch.tensor(special_tokens_mask2, dtype=torch.bool)
        special_tokens_mask = sum(labels == i
                                  for i in self.exclude_tokens).bool()

        # print((t2-t1)*1000)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer),
                                     labels.shape,
                                     dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
