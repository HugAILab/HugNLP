# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 9:13 下午
# @Author  : JianingWang
# @File    : data_collator

import torch
from dataclasses import dataclass
from typing import Optional
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForGlobalPointer:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    pad_to_max_length: Optional[bool] = None

    def __call__(self, features):
        # Tokenize
        is_train = features[0]["is_train"] > 0
        batch = []
        for f in features:
            batch.append({"input_ids": f["input_ids"],
                          "token_type_ids": f["token_type_ids"],
                          "attention_mask": f["attention_mask"]})
        batch = self.tokenizer.pad(
            batch,
            padding="max_length",  # 为了index不出错直接Padding到max length，如果用longest，后面的np.unravel_index也要改
            max_length=self.max_length,
            return_tensors="pt"
        )
        # 确定label
        if not is_train:
            return batch
        else:
            # label之所以这样设置，是为了适应于多区间阅读理解任务（多标签分类）
            labels = torch.zeros(len(features), 1, self.max_length, self.max_length)  # 阅读理解任务entity种类为1 [bz, 1, max_len, max_len]
            for feature_id, feature in enumerate(features): # 遍历每个样本
                starts, ends = feature["start"], feature["end"]
                offset = feature["offset_mapping"] # 表示tokenizer生成的token对应原始文本中字符级别的位置区间
                position_map = {}
                for i, (m, n) in enumerate(offset):
                    if i != 0 and m == 0 and n == 0:
                        continue
                    for k in range(m, n + 1):
                        position_map[k] = i # 字符级别的第k个字符属于分词i
                for start, end in zip(starts, ends):
                    end -= 1
                    # MRC 没有答案时则把label指向CLS
                    if start == 0:
                        assert end == -1
                        labels[feature_id, 0, 0, 0] = 1
                    else:
                        if start in position_map and end in position_map:
                            # 指定下列元素为1，说明表示第feature_id个样本的预测区间
                            labels[feature_id, 0, position_map[start], position_map[end]] = 1

            # short_labels没用，解决transformers trainer默认会merge labels导致内存爆炸的问题
            # 需配合--label_names=short_labels使用
            batch["labels"] = labels
            if batch["labels"].max() > 0:
                batch["short_labels"] = torch.ones(len(features))
            else:
                batch["short_labels"] = torch.zeros(len(features))
            return batch
