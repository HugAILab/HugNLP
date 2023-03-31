# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 9:13 下午
# @Author  : JianingWang
# @File    : data_process
import enum
import json
# from random import random
import random
import torch
import os.path
import numpy as np
from dataclasses import dataclass
from datasets import DatasetDict, Dataset, load_metric
from typing import Optional, List
from collections import defaultdict
from transformers import PreTrainedTokenizerBase
from processors.benchmark.clue.utils import InputExample

from processors.ProcessorBase import CLSProcessor
from metrics import datatype2metrics
from collections import defaultdict, Counter
from processors.basic_processors.prompt_processor import InstructionPromptProcessor
from tools.processing_utils.tokenizer.tokenizer_utils import get_special_token_mapping
from processors.instruction_prompting.incontext_learning.data_collator import DataCollatorForClassificationInContextLearning
from processors.dataset import DatasetK
from tools.model_utils.gpt_response import GPTResponse

"""
Causal LM for classification in-context learning
"""
class CausalInContextClassificationProcessor(CLSProcessor):
    def __init__(self,
        data_args,
        training_args,
        model_args,
        tokenizer=None,
        post_tokenizer=False,
        keep_raw_data=True):

        super().__init__(data_args,
                         training_args,
                         model_args,
                         tokenizer,
                         post_tokenizer=post_tokenizer,
                         keep_raw_data=keep_raw_data)
        param = {
            p.split("=")[0]: p.split("=")[1]
            for p in (data_args.user_defined).split(" ")
        }
        self.data_name = param["data_name"] if "data_name" in param.keys() else "user-define"
        self.num_incontext_example = int(param["num_incontext_example"]) # the number of in-context example
        self.l = int(param["l"]) if "l" in param.keys() else 1 # the max length to generate
        self.use_calibrate = param["use_calibrate"] == "True" if "use_calibrate" in param.keys() else False # whether to calibrate the prediction
        self.content_free = ["N/A"] # When set "use_calibrate" is True, will add a new content_free example.
        self.data_dir = data_args.data_dir
        self.train_file = os.path.join(
            data_args.data_dir, "train.json"
        )  # each line: {"sentence1": xx, "sentence2": xx, "label": xx}

        self.dev_file = os.path.join(data_args.data_dir, "dev.json")
        self.test_file = os.path.join(data_args.data_dir, "test.json")
        assert os.path.exists(self.test_file), "the test.json is not found in {}".format(self.data_dir)
        # each line is one label name
        self.label_file = os.path.join(data_args.data_dir,"label_names.json")
        self.template_file = os.path.join(data_args.data_dir, "template.json")
        self.instruction_file = os.path.join(data_args.data_dir, "instruction.json")
        self.label_words_mapping_file = os.path.join(data_args.data_dir, "label_words_mapping.json")
        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.sentence1_key = "sentence1"
        self.sentence2_key = "sentence2"

        # 如果用户输入了label name，则以用户输入的为准
        if "label_names" in param.keys():
            self.labels = param["label_names"].replace(" ", "").split(",")
        # 如果用户没有输入label name，检查本地是否有label_names.json文件
        elif os.path.exists(self.label_file): # {"label_name": "description", xxx}
            self.labels = list()
            with open(self.label_file, "r", encoding="utf-8") as fr:
                label_name_dict = json.load(fr)
            self.labels = list(label_name_dict.keys())
        else:
            raise FileNotFoundError(
                "You must define the 'label_names' in user-define parameters or"
                "define a label_names.json file at {}".format(self.label_file))

        self.label2id = {label: ei for ei, label in enumerate(self.labels)}
        self.id2label = {ei: label for ei, label in enumerate(self.labels)}


        assert self.model_args.use_prompt_for_cls == True, "If you want to perform classification by in-context learning, you must add the parameter 'use_prompt_for_cls' in config"

        """
        template:
        [{
            "prefix_template": "",
            "suffix_template": "This is <mask> ."
        }, None]

        label_words_mapping:
        {
            "unacceptable": ["incorrect"],
            "acceptable": ["correct"]
        }
        """
        assert os.path.exists(self.template_file) and os.path.exists(self.label_words_mapping_file), "If you want to use prompt, you must add two files ({} and {}).".format(self.template_file, self.label_words_mapping_file)
        template = json.load(open(self.template_file, "r", encoding="utf-8"))
        instruction = json.load(open(self.instruction_file, "r", encoding="utf-8"))
        # label_words_mapping: {"xxx": ["xxx"], ...}
        self.label_words_mapping = json.load(open(self.label_words_mapping_file, "r", encoding="utf-8"))

        if self.model_args.model_type in ["gpt2"]:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.incontext_examples = None
        self.prompt_engineering = InstructionPromptProcessor(
            data_args=self.data_args,
            task_name=self.data_name,
            tokenizer=self.tokenizer,
            sentence1_key=self.sentence1_key,
            sentence2_key=self.sentence2_key,
            template=template,
            instruction=instruction,
            label_words_mapping=self.label_words_mapping)

        self.label_word_list = self.prompt_engineering.obtain_label_word_list()

        # define for api response
        self.api_response = None
        if model_args.model_type == "gpt2":
            self.api_response = GPTResponse("gpt2", data_path=self.data_dir) # default for gpt2

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token # GPT需要显式地添加padding token

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForClassificationInContextLearning(self.tokenizer, pad_to_multiple_of=8 if pad_to_multiple_of_8 else None, pad_to_max_length=self.data_args.pad_to_max_length)

    def InContextSampling(self, examples: list) -> list:
        # used for sampling in-context examples
        random.shuffle(examples)
        incontext_examples = examples[:self.num_incontext_example] if self.num_incontext_example < len(examples) else examples
        return incontext_examples


    def get_tokenized_datasets(self):
        raw_datasets = DatasetDict()
        if self.training_args.do_train:
            train_examples = self.get_examples("train")
            raw_datasets["train"] = DatasetK.from_dict(self.list_2_json(train_examples)) # [{k1: xxx, k2: xxx}, {...}] -> {k1: [xxx, xxx], k2: [xxx, xxx]}
        if self.training_args.do_eval:
            dev_examples = self.get_examples("dev")
            raw_datasets["validation"] = DatasetK.from_dict(self.list_2_json(dev_examples))
            print("raw_datasets[validation][0]=", raw_datasets["validation"][0])
        if self.training_args.do_predict:
            test_examples = self.get_examples("test")
            raw_datasets["test"] = DatasetK.from_dict(self.list_2_json(test_examples))

        if self.post_tokenizer:
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets
            return raw_datasets
        remove_columns = self.sentence1_key if not self.sentence2_key else [self.sentence1_key, self.sentence2_key]
        tokenize_func = self.build_preprocess_function()
        # 多gpu, 0计算完存cache，其他load cache
        with self.training_args.main_process_first(desc="dataset tokenizer map"):
            raw_datasets = raw_datasets.map(
                tokenize_func,
                batched=True,
                desc="Running tokenizer on dataset",
                remove_columns=remove_columns
            )
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets
            return raw_datasets

    def get_content_free_examples(self):
        assert self.use_calibrate and self.content_free is not None and self.incontext_examples is not None
        content_free_examples = list()
        for ei, content_free_text in enumerate(self.content_free):
            content_free_example = {
                self.sentence1_key: content_free_text,
            }
            content_free_prompt = self.prompt_engineering.construct_incontext_prompt(
                sentence1_key=self.sentence1_key,
                sentence2_key=self.sentence2_key,
                incontext_examples=self.incontext_examples,
                eval_example=content_free_example
            )
            content_free_examples.append({
                "idx": "content-free-{}".format(ei),
                "content_free_prompt": content_free_prompt,
                # "label": 0, # content free example has no label (not bias to any labels). But is must add a label value, so set 0.
                # "target": 0, # content free example has no label (not bias to any labels). But is must add a label value, so set 0.
            })
        return content_free_examples

    def get_examples(self, set_type):
        # assert set_type != "train", "In-context learning dose not have training proce"

        examples = list()

        if set_type == "train":
            # 仅用于支持本框架（默认必须加载训练集）
            examples = self._create_examples(self._read_json2(self.train_file), "train")
            return examples # List[InputExample]
        else:# dev或test时
            # 先获取所有的训练集
            training_examples = self._create_examples(self._read_json2(self.train_file), "train")
            # 随机采样若干in-context example作为demonstration
            if self.incontext_examples is None:
                self.incontext_examples = self.InContextSampling(training_examples)
            # incontext_examples = self.incontext_examples
            if set_type == "dev":
                examples = self._create_examples(self._read_json2(self.dev_file), set_type)
            else:
                examples = self._create_examples(self._read_json2(self.test_file), set_type)
            # 为每个dev/test构建prompt
            eval_examples = list()
            for example in examples:
                prompt = self.prompt_engineering.construct_incontext_prompt(
                    sentence1_key=self.sentence1_key,
                    sentence2_key=self.sentence2_key,
                    incontext_examples=self.incontext_examples,
                    eval_example=example
                    )
                eval_examples.append({
                    "idx": example["idx"],
                    self.sentence1_key: prompt,
                    self.sentence2_key: "",
                    "label": example["label"],
                    "target": example["target"],
                })

        return eval_examples # List[dict]

    def _create_examples(self, lines, set_type=None):
        examples = []
        is_train = 0 if set_type == "test" else 1
        for idx, line in enumerate(lines):
            sentence1 = line[self.sentence1_key]
            sentence2 = line[self.sentence2_key] if self.sentence2_key in line.keys() else None
            if set_type != "test":
                label = line["label"]
                if label not in self.labels:
                    continue
            else:
                # 有些测试集没有标签，为了避免报错，默认初始化标签0
                label = line["label"] if "label" in line.keys() else self.labels[0]

            examples.append({
                "idx": idx,
                self.sentence1_key: sentence1,
                self.sentence2_key: sentence2,
                "label": self.label2id[label], # label id
                "target": label # label name
            })

        return examples

    def set_config(self, config):
        pass

    def build_preprocess_function(self):
        # Tokenize the texts
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length

        def func(examples):
            # Tokenize
            tokenized_examples = tokenizer(
                examples[self.sentence1_key], # 即使是sentence pair任务，也在数据处理前通过prompt合并为一个序列
                truncation=True,
                # max_length=max_seq_length,
                # padding="max_length" if self.data_args.pad_to_max_length else False,
                # return_offsets_mapping=True
            )
            # 确定label
            return tokenized_examples

        return func
