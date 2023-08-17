'''
# -*- coding: utf-8 -*-
Author: nchen909 NuoChen
Date: 2023-05-06 16:11:10
FilePath: /HugNLP/processors/benchmark/codexglue/data_collator.py
'''
import torch
from dataclasses import dataclass
from typing import Optional
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForCodeXGLUE:
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
            if "token_type_ids" in f.keys():
                batch.append(
                    {
                        "input_ids": f["input_ids"],
                        "token_type_ids": f["token_type_ids"],
                        "attention_mask": f["attention_mask"]
                    }, )
            else:
                batch.append(
                    {
                        "input_ids": f["input_ids"],
                        "attention_mask": f["attention_mask"]
                    }, )
        batch = self.tokenizer.pad(
            batch,
            padding=
            "max_length",  # 为了index不出错直接Padding到max length，如果用longest，后面的np.unravel_index也要改
            max_length=self.max_length,
            return_tensors="pt")  # {"input_ids": [xxx], xxx}
        # add position_ids

        # add labels
        batch["labels"] = torch.Tensor([int(f["label"]) for f in features]).long()
        # add mask_pos (when using prompt-tuning, need to record the masked position for each input_ids)

        if "mask_pos" in features[0].keys():
            batch["mask_pos"] = torch.Tensor([f["mask_pos"] for f in features]).long()

        return batch
