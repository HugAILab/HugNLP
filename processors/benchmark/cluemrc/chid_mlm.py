# -*- coding: utf-8 -*-
# @Time    : 2022/4/18 12:02 a.m.
# @Author  : JianingWang
# @File    : chid_mlm.py

import json
import re
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Union, Optional

import numpy as np

from processors.ProcessorBase import CLSProcessor
from transformers import PreTrainedTokenizerBase, EvalPrediction
from transformers.file_utils import PaddingStrategy


@dataclass
class ChidMLMDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    anchor_token_id: Optional[int] = None

    def cut_text(self, text_input):
        max_length = self.max_length
        if len(text_input) <= max_length - 2:
            return text_input
        a_idx = text_input.index(self.anchor_token_id)
        head, tail, a_token = text_input[:a_idx], text_input[a_idx + 1:], text_input[a_idx:a_idx + 1]
        mlen = max_length - 3
        if len(head) < mlen // 2:
            tail = tail[:mlen - len(head)]
        elif len(tail) < mlen // 2:
            head = head[-(mlen - len(tail)):]
        else:
            head, tail = head[- mlen // 2:], tail[:mlen // 2]
        return head + a_token + tail

    def __call__(self, features):
        labels = [feature.pop("label") for feature in features] if "label" in features[0] else None
        text = [feature.pop("text_a") for feature in features]
        batch = self.tokenizer(
            text,
            add_special_tokens=False, return_token_type_ids=False, return_attention_mask=False
        )
        batch["input_ids"] = [self.cut_text(input_id) for input_id in batch["input_ids"]]
        batch["token_type_ids"] = [[0] * len(input_id) for input_id in batch["input_ids"]]
        batch["attention_mask"] = [[1] * len(input_id) for input_id in batch["input_ids"]]
        if labels is not None:
            batch["labels"] = labels
        batch = self.tokenizer.pad(
            batch,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return batch


class ChidMLMProcessor(CLSProcessor):
    def __init__(self, data_args, training_args, model_args):
        super().__init__(data_args, training_args, model_args, post_tokenizer=True)
        self.data_args = data_args
        self.train_file = os.path.join(data_args.data_dir, "train.json")
        self.train_answer_file = os.path.join(data_args.data_dir, "train_answer.json")
        self.dev_file = os.path.join(data_args.data_dir, "dev.json")
        self.dev_answer_file = os.path.join(data_args.data_dir, "dev_answer.json")
        idiom_dict_file = os.path.join(data_args.data_dir, "idiomDict.json")
        idiom_meaning = self._read_json(idiom_dict_file)
        idiom_list = list(idiom_meaning.keys())
        idiom_list.sort()
        self.labels = idiom_list
        self.label_to_id = {l: i for i, l in enumerate(self.labels)}
        self.tag_token = "[IDIOM]"
        self.test_idiom = []

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return ChidMLMDataCollator(self.tokenizer,
                                   padding="longest",
                                   max_length=self.data_args.max_seq_length,
                                   pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
                                   anchor_token_id=self.tag_token_id)

    def get_examples(self, set_type):
        if set_type == "train":
            train_answer = self._read_json(self.train_answer_file)
            examples = self._create_examples(self._read_jsonl(self.train_file), train_answer, "train")
            self.train_examples = examples
        elif set_type == "dev":
            dev_answer = self._read_json(self.dev_answer_file)
            examples = self._create_examples(self._read_jsonl(self.dev_file), dev_answer, "dev")
            self.dev_examples = examples
        elif set_type == "test":
            train_answer = self._read_json(self.train_answer_file)
            examples = self._create_examples(self._read_jsonl(self.train_file), train_answer, "test")
            self.test_examples = examples
        return examples

    def _create_examples(self, lines, true_label_map, set_type):
        examples = []

        for line in lines:
            candidates = line["candidates"]
            for text in line["content"]:
                text = text.replace("“", """)
                text = text.replace("”", """)
                if text.endswith("..."):
                    text = text[:-3]
                idioms = re.findall("#idiom\\d{6}#", text)
                for idiom in idioms:
                    if set_type == "test":
                        self.test_idiom.append(idiom)
                    new_text = deepcopy(text)
                    # 将不需要预测的idiom转为[MASK][MASK][MASK][MASK]
                    for o in idioms:
                        if o != idiom:
                            new_text = new_text.replace(o, "[MASK][MASK][MASK][MASK]")

                    text_a = new_text.replace(idiom, "[IDIOM]")
                    label = self.label_to_id[candidates[true_label_map[idiom]]]
                    example = {"text_a": text_a, "label": label}
                    examples.append(example)

        print(len(examples))
        print(examples[0])
        return examples

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.add_tokens(self.tag_token)
        self.tag_token_id = self.tokenizer.convert_tokens_to_ids(self.tag_token)

    def set_config(self, config):
        config.tag_token_id = self.tag_token_id

    def compute_metrics(self, p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        acc = (preds == labels).mean()

        return {
            "eval_acc": round(acc, 4),
            "eval_score": round(acc, 4)
        }

    def save_result(self, logits, labels):
        top20 = logits.argsort()[:, -20:][:, ::-1].tolist()
        out = {}
        for idiom, candi in zip(self.test_idiom, top20):
            idiom_20 = [self.labels[c] for c in candi]
            out[idiom] = idiom_20
        outfile = os.path.join(self.data_args.data_dir, "c3_train_mlm_top20.json")
        with open(outfile, "w", encoding="utf8") as f:
            json.dump(out, f, indent=4, ensure_ascii=False)
