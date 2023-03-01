# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 4:36 下午
# @Author  : JianingWang
# @File    : data_collator.py
import torch
from typing import Optional
from dataclasses import dataclass
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast


@dataclass
class DataCollatorForCausalLM:
    tokenizer: GPT2TokenizerFast
    max_length: Optional[int] = 1024
    pad_to_max_length: Optional[bool] = None

    def __call__(self, features):
        # Tokenize
        # is_train = features[0]['is_train'] > 0
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        batch = []
        for f in features:
            input_ids = [self.tokenizer.bos_token_id
                         ] + f['input_ids'] + [self.tokenizer.eos_token_id]
            labels = input_ids
            # label_masks = [0] + f["label_masks"] + [0]
            if 'attention_mask' in f.keys():
                attention_mask = [1] + f['attention_mask'] + [0]
            else:
                attention_mask = [1] + [1] * len(f['input_ids']) + [0]

            if 'token_type_ids' in f.keys():
                token_type_ids = [1] + f['token_type_ids'] + [0]
            else:
                token_type_ids = [0] + [0] * len(f['input_ids']) + [0]

            num_padding = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * num_padding
            labels += [-100] * num_padding

            # label_masks += [0] * num_padding
            attention_mask += [0] * num_padding
            token_type_ids += [0] * num_padding

            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

            # label_masks = label_masks[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            token_type_ids = token_type_ids[:self.max_length]

            assert len(input_ids) == len(labels)
            assert len(input_ids) == len(attention_mask)
            assert len(input_ids) == len(token_type_ids)

            batch.append({
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                # 'label_masks': label_masks,
            })
        batch = {
            key: torch.tensor([feature[key] for feature in batch],
                              dtype=torch.long)
            for key in batch[0].keys()
        }
        return batch
