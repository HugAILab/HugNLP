# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 9:13 p.m.
# @Author  : JianingWang
# @File    : data_collator.py

import torch
from dataclasses import dataclass
from typing import Optional
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollator:
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
            batch.append({"input_ids": f["input_ids"],
                          "token_type_ids": f["token_type_ids"],
                          "attention_mask": f["attention_mask"]},
                          )
        batch = self.tokenizer.pad(
            batch,
            padding="max_length",  # 为了index不出错直接Padding到max length，如果用longest，后面的np.unravel_index也要改
            max_length=self.max_length,
            return_tensors="pt"
        ) # {"input_ids": [xxx], xxx}
        # add position_ids

        # add labels
        batch["labels"] = torch.Tensor([f["label"] for f in features]).long()

        # add by wjn: 获得每个example每个segment的区间
        if self.is_segment_spans:
            segment_spans = list()
            token_type_ids = batch["token_type_ids"]
            for token_type_id in token_type_ids:
                if torch.sum(token_type_id) == 0: # 说明只有一个segment
                    break
                seg1_start = 1
                seg1_end = -1
                seg2_start = -1
                seg2_end = -1
                for ei, token_type in enumerate(token_type_id):
                    if token_type == 1 and seg1_end == -1:
                        seg1_end = ei - 2
                        seg2_start = ei
                        continue
                    if token_type == 0 and seg1_end != -1:
                        seg2_end = ei - 2
                        break
                segment_spans.append([seg1_start, seg1_end, seg2_start, seg2_end])
            if len(segment_spans) != 0:
                batch["segment_spans"] = torch.Tensor(segment_spans).long()
        return batch
