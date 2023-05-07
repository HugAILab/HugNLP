# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 5:50 p.m.
# @Author  : JianingWang
# @File    : data_collator.py

import torch
from dataclasses import dataclass
from typing import Optional
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForDefaultPairwiseRewardTraining:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    pad_to_max_length: Optional[bool] = None
    is_segment_spans: Optional[bool] = False

    def __call__(self, features):
        # Tokenize
        # is_train = features[0]["is_train"] > 0
        batch = []
        for f in features:

            chosen_sequences = f["chosen_sequences"] + [self.tokenizer.pad_token_id] * (self.max_length - len(f["chosen_sequences"]))
            chosen_attention_mask = f["chosen_attention_mask"] + [0] * (self.max_length - len(f["chosen_attention_mask"]))
            rejected_sequences = f["rejected_sequences"] + [self.tokenizer.pad_token_id] * (self.max_length - len(f["rejected_sequences"]))
            rejected_attention_mask = f["rejected_attention_mask"] + [0] * (self.max_length - len(f["rejected_attention_mask"]))

            # print("chosen_sequences=", chosen_sequences)
            # print("chosen_attention_mask=", chosen_attention_mask)
            # print("rejected_sequences=", rejected_sequences)
            # print("rejected_attention_mask=", rejected_attention_mask)
            
            batch.append({
                "chosen_sequences": chosen_sequences,
                "chosen_attention_mask": chosen_attention_mask,
                "rejected_sequences": rejected_sequences,
                "rejected_attention_mask": rejected_attention_mask,
            })

        # batch = self.tokenizer.pad(
        #     batch,
        #     padding="max_length",
        #     max_length=self.max_length,
        #     return_tensors="pt"
        # ) 
        batch = {key: torch.Tensor([f[key] for f in batch]).long() for key in batch[0].keys()}

        return batch