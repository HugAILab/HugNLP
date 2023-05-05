# -*- coding: utf-8 -*-
# @Time    : 2022/2/27 12:58 p.m.
# @Author  : JianingWang
# @File    : c3

import json
import re
import os
from dataclasses import dataclass
from typing import Optional, Union, List

import numpy as np
from itertools import chain
from copy import deepcopy

import torch
from datasets import load_from_disk
from transformers import EvalPrediction, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from processors.ProcessorBase import DataProcessor, CLSProcessor
from data.data_collator import DataCollatorForMultipleChoice


@dataclass
class PostDataCollatorTagForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    max_eval_seq_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    anchor_token_id: int = None
    tag_token_id: List[int] = None
    is_train: bool = True
    idiom_meaning: Optional[dict] = None
    meaning_len: Optional[int] = 0
    pseudo_train: Optional[bool] = False

    def cut_text(self, text_input):
        # 输入原始的一个文本，根据需要填空的位置，将文本分为head和tail两段
        max_length = self.max_length - self.meaning_len if self.is_train else self.max_eval_seq_length - self.meaning_len
        a_idx = text_input.index(self.anchor_token_id)
        head, tail, a_token = text_input[:a_idx], text_input[a_idx + 1:], text_input[a_idx:a_idx + 1]

        if len(text_input) <= max_length - 7:
            return head, tail
        mlen = max_length - 8
        if len(head) < mlen // 2:
            tail = tail[:mlen - len(head)]
        elif len(tail) < mlen // 2:
            head = head[-(mlen - len(tail)):]
        else:
            head, tail = head[- mlen // 2:], tail[:mlen // 2]
        return head, tail

    def __call__(self, features):
        if not features[0]["is_train"]:
            self.is_train = False
        else:
            self.is_train = True
        batch_size = len(features)
        num_choices = len(features[0]["candidates"])
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features] if label_name in features[0] else None
        text = [feature.pop("text_a") for feature in features]
        candidates = [feature.pop("candidates") for feature in features]
        text_batch = self.tokenizer(text, add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)
        candi_batch = self.tokenizer(list(chain(*candidates)), add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False)
        # 拼接成语的文字描述
        if self.idiom_meaning:
            meaning = [self.idiom_meaning[candi] for candi in list(chain(*candidates))]
            meaning_batch = self.tokenizer(meaning, truncation=True, max_length=self.meaning_len, add_special_tokens=False, return_token_type_ids=False,
                                           return_attention_mask=False)
        flattened_features = []
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        for i in range(batch_size):
            head, tail = self.cut_text(text_batch["input_ids"][i])
            for j in range(i * num_choices, (i + 1) * num_choices):
                candi_input = candi_batch["input_ids"][j] # 第j个候选成语的inputid
                assert len(candi_input) == 4
                # [CLS] xxxxxxxxx <start> xx <end> xxxxxx. [SEP]
                input_id = [cls_token_id] + head + [self.tag_token_id[0]] + candi_input + [self.tag_token_id[1]] + tail + [sep_token_id]
                token_type_id = [0] * len(input_id)
                # 拼接成语的文字描述
                if self.idiom_meaning:
                    meaning_input = meaning_batch["input_ids"][j] # 第j个成语的解释文本inputid
                    input_id = input_id + meaning_input + [sep_token_id]
                    token_type_id = token_type_id + [1] * (len(meaning_input) + 1)
                attention_mask = [1] * len(input_id)
                flattened_features.append({"input_ids": input_id, "token_type_ids": token_type_id, "attention_mask": attention_mask})

        batch = self.tokenizer.pad(
            flattened_features,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()} # [batch_size, num_choices]
        # Add back labels
        if labels is None:
            return batch

        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        if self.pseudo_train:
            pseudo = [feature.pop("pseudo") for feature in features]
            batch["pseudo"] = torch.tensor(pseudo, dtype=torch.int64)
        return batch


class ChidTagProcessor(CLSProcessor):

    def __init__(self, data_args, training_args, model_args):
        super().__init__(data_args, training_args, model_args, post_tokenizer=True)
        self.train_file = os.path.join(data_args.data_dir, "train.json")
        self.train_answer_file = os.path.join(data_args.data_dir, "train_answer.json")
        self.dev_file = os.path.join(data_args.data_dir, "dev.json")
        self.dev_answer_file = os.path.join(data_args.data_dir, "dev_answer.json")
        self.test_file = os.path.join(data_args.data_dir, "test1.1.json")
        idiom_dict_file = os.path.join(data_args.data_dir, "idiomDict.json")
        # self.hard_negative = self._read_json(os.path.join(data_args.data_dir, "c3_train_mlm_top20.json"))
        self.idiom_meaning = self._read_json(idiom_dict_file)
        self.tag_tokens = ["<start>", "<end>"] # 区间标记
        self.anchor_token = "<idiom>" # 表示需要填空的位置
        self.add_meaning = False
        self.add_had_negative = False
        self.pseudo_train = False
        self.test_idiom = [] # 保存测试集中要预测的位置

    def get_data_collator(self): # 回调函数，用于huggingface的dataloader
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        max_eval_seq_length = self.data_args.max_eval_seq_length if self.data_args.max_eval_seq_length else self.data_args.max_seq_length
        kwargs = {
            "tokenizer": self.tokenizer,
            "padding": "longest",
            "max_length": self.data_args.max_seq_length,
            "max_eval_seq_length": max_eval_seq_length,
            "pad_to_multiple_of": 8 if pad_to_multiple_of_8 else None,
            "anchor_token_id": self.anchor_token_id,
            "tag_token_id": self.tag_token_ids,
            "pseudo_train": self.pseudo_train
        }
        if self.add_meaning:
            kwargs["idiom_meaning"] = self.idiom_meaning
            kwargs["meaning_len"] = 30
        return PostDataCollatorTagForMultipleChoice(**kwargs)

    def get_examples(self, set_type):
        if set_type == "train":
            train_answer = self._read_json(self.train_answer_file) # label (正确答案在candidate中的序号)
            examples = self._create_examples(self._read_jsonl(self.train_file), train_answer, "train") # input
            if self.pseudo_train:
                dev_answer = self._read_json(self.dev_answer_file)
                pse1 = self._create_examples(self._read_jsonl(self.dev_file), dev_answer, "train")
                # predict_file = "/Users/JianingWang/预训练模型/任务/CLUEMRC/outputs/submit_20220419/chid11_predict.json"
                predict_file = os.path.join(self.data_args.data_dir, "chid11_predict.json")
                pseudo_answer = self._read_json(predict_file)
                pse2 = self._create_examples(self._read_jsonl(self.test_file), pseudo_answer, "train", pseudo=True)
                examples = examples + pse1 + pse2
            self.train_examples = examples
        elif set_type == "dev":
            dev_answer = self._read_json(self.dev_answer_file)
            examples = self._create_examples(self._read_jsonl(self.dev_file), dev_answer, "dev")
            self.dev_examples = examples
        elif set_type == "test":
            examples = self._create_examples(self._read_jsonl(self.test_file), {}, "test")
            self.test_examples = examples
        return examples

    def _create_examples(self, lines, true_label_map, set_type, pseudo=False):
        examples = []
        for qid, line in enumerate(lines):
            candidates = line["candidates"] # 候选的成语列表（10个）
            for text in line["content"]: # 正文
                text = text.replace("“", """)
                text = text.replace("”", """)
                if text.endswith("..."):
                    text = text[:-3]
                idioms = re.findall("#idiom\\d{6}#", text) # 寻找所有需要填空的位置
                for idiom in idioms: # 遍历每一个填空，每一个空当作一个训练样本。idiom是当前指定要预测的位置
                    if set_type == "test":
                        self.test_idiom.append(idiom)
                    new_text = deepcopy(text)
                    # 将不需要预测的idiom转为[MASK][MASK][MASK][MASK]
                    for o in idioms:
                        if o != idiom:
                            new_text = new_text.replace(o, "[MASK][MASK][MASK][MASK]")
                    example = {
                        "text_a": new_text.replace(idiom, self.anchor_token),
                        "candidates": deepcopy(candidates),
                        "label": true_label_map.get(idiom, 0),
                        "qid": qid,
                        "pseudo": 1 if pseudo else 0
                    }
                    if set_type == "train":
                        example["is_train"] = True
                    else:
                        example["is_train"] = False
                    examples.append(example)

        return examples

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.add_tokens(self.anchor_token)
        self.anchor_token_id = self.tokenizer.convert_tokens_to_ids(self.anchor_token)
        self.tokenizer.add_tokens(self.tag_tokens)
        self.tag_token_ids = self.tokenizer.convert_tokens_to_ids(self.tag_tokens)

    def set_config(self, config):
        config.is_relation_task = True
        config.start_token_ids = self.tag_token_ids


    def compute_metrics(self, p: EvalPrediction):
        from scipy.optimize import linear_sum_assignment
        c, n = 0, [0]
        for i in self.dev_examples:
            if i["qid"] == c:
                n[-1] += 1
            else:
                c = i["qid"]
                n.append(n[-1] + 1)
        preds = []
        for predict in np.split(p.predictions, n):
            pred = linear_sum_assignment(-predict)[1]
            preds.extend(pred)

        labels = p.label_ids
        try:
            acc = (preds == labels).mean()
        except:
            acc = 0.0
        return {
            "eval_acc": round(acc, 4)
        }

    def save_result(self, logits, labels):
        from scipy.optimize import linear_sum_assignment
        c, n = 0, [0]
        for i in self.test_examples:
            if i["qid"] == c:
                n[-1] += 1
            else:
                c = i["qid"]
                n.append(n[-1] + 1)
        preds = []
        for predict in np.split(logits, n):
            pred = linear_sum_assignment(-predict)[1]
            preds.extend(pred)
        predicts = {k: int(v) for k, v in zip(self.test_idiom, preds)}
        outfile = os.path.join(self.training_args.output_dir, "test1.1_pred.json")
        with open(outfile, "w", encoding="utf8") as f:
            json.dump(predicts, f, indent=4, ensure_ascii=False)
