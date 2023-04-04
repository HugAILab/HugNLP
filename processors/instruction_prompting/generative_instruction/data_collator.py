# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 9:13 下午
# @Author  : JianingWang
# @File    : data_collator.py
import torch
from typing import Optional
from dataclasses import dataclass
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from collections.abc import Mapping

@dataclass
class DataCollatorForCausalLM:
    tokenizer: GPT2TokenizerFast
    max_length: Optional[int] = 1024
    pad_to_max_length: Optional[bool] = None

    def __call__(self, features):
        # Tokenize
        # is_train = features[0]["is_train"] > 0
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
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
        if "labels" in features[0].keys():
            batch["labels"] = torch.Tensor([f["labels"] for f in features]).long()
        else:
            batch["labels"] = batch["input_ids"].copy()
        # add mask_pos (when using prompt-tuning, need to record the masked position for each input_ids)

        if "mask_pos" in features[0].keys():
            batch["mask_pos"] = torch.Tensor([f["mask_pos"] for f in features]).long()

        return batch

    # def __call__(self, features):
    #     # Tokenize
    #     # is_train = features[0]["is_train"] > 0
    #     self.tokenizer.pad_token = self.tokenizer.eos_token
    #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    #     batch = []
    #     for f in features:
    #         input_ids = f["input_ids"]
    #         labels = f["labels"] if "labels" in f.keys() else input_ids.copy()
    #         # label_masks = [0] + f["label_masks"] + [0]
    #         if "attention_mask" in f.keys():
    #             attention_mask = f["attention_mask"]
    #         else:
    #             attention_mask = [1] * len(f["input_ids"])

    #         if "token_type_ids" in f.keys():
    #             token_type_ids = f["token_type_ids"]
    #         else:
    #             token_type_ids = [0] * len(f["input_ids"])

    #         num_padding = self.max_length - len(input_ids)
    #         input_ids += [self.tokenizer.pad_token_id] * num_padding
    #         labels += [self.tokenizer.pad_token_id] * num_padding

    #         # label_masks += [0] * num_padding
    #         attention_mask += [0] * num_padding
    #         token_type_ids += [0] * num_padding

    #         input_ids = input_ids[:self.max_length]
    #         labels = labels[:self.max_length]

    #         # label_masks = label_masks[:self.max_length]
    #         attention_mask = attention_mask[:self.max_length]
    #         token_type_ids = token_type_ids[:self.max_length]

    #         assert len(input_ids) == len(labels)
    #         assert len(input_ids) == len(attention_mask)
    #         assert len(input_ids) == len(token_type_ids)

    #         batch.append({
    #             "input_ids": input_ids,
    #             "labels": labels,
    #             "attention_mask": attention_mask,
    #             "token_type_ids": token_type_ids,
    #             # "label_masks": label_masks,
    #             })
    #     batch = {key: torch.tensor([feature[key] for feature in batch], dtype=torch.long) for key in batch[0].keys()}
    #     return batch
