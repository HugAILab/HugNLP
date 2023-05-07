# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 5:50 p.m.
# @Author  : JianingWang
# @File    : data_processor.py

import json
import torch
import os.path
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from datasets import DatasetDict, Dataset, load_metric
from processors.dataset import DatasetK
from processors.ProcessorBase import CLSProcessor
from processors.benchmark.clue.clue_processor import clue_processors, clue_output_modes
from metrics import datatype2metrics
from tools.computations.softmax import softmax
from processors.reinforcement_learning.data_collator import DataCollatorForDefaultPairwiseRewardTraining
from processors.basic_processors.prompt_processor import PromptBaseProcessor
from tools.processing_utils.tokenizer.tokenizer_utils import get_special_token_mapping


"""
The data processor for the pair-wise reward in reinforcement learning.
"""
class PairwiseRewardProcessor(CLSProcessor):
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
            for p in (data_args.user_defined).split(" ") if p != ""
        }
        self.data_name = param["data_name"] if "data_name" in param.keys() else "preference data"
        self.data_dir = data_args.data_dir
        self.train_file = os.path.join(
            data_args.data_dir, "preference_train.json"
        )  # each line: {"sentence1": xx, "sentence2": xx, "label": xx}
        self.dev_file = os.path.join(data_args.data_dir, "preference_dev.json")
        self.test_file = os.path.join(data_args.data_dir, "preference_test.json")
        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.sentence1_key = "chosen"
        self.sentence2_key = "rejected"


    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForDefaultPairwiseRewardTraining(
            self.tokenizer,
            max_length=self.data_args.max_seq_length,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            pad_to_max_length=self.data_args.pad_to_max_length)

    def get_examples(self, set_type):
        examples = list()
        if set_type == "train":
            examples = self._create_examples(self._read_json2(self.train_file), "train")
            self.train_examples = examples
        elif set_type == "dev":
            examples = self._create_examples(self._read_json2(self.dev_file), "dev")
            self.dev_examples = examples
        elif set_type == "test":
            examples = self._create_examples(self._read_json2(self.test_file), "test")
            self.test_examples = examples
        return examples

    def _create_examples(self, lines, set_type):
        examples = list()
        for ei, line in enumerate(lines):
            idx = "{}-{}".format(set_type, str(ei))
            chosen_sequence = line[self.sentence1_key]
            rejected_sequence = line[self.sentence2_key]

            examples.append({
                "idx": idx,
                self.sentence1_key: chosen_sequence,
                self.sentence2_key: rejected_sequence,
            })

        return examples

    def get_tokenized_datasets(self):
        raw_datasets = DatasetDict()
        if self.training_args.do_train:
            train_examples = self.get_examples("train")
            raw_datasets["train"] = DatasetK.from_dict(self.list_2_json(train_examples))  # [{k1: xxx, k2: xxx}, {...}] -> {k1: [xxx, xxx], k2: [xxx, xxx]}
        if self.training_args.do_eval:
            dev_examples = self.get_examples("dev")
            raw_datasets["validation"] = DatasetK.from_dict(self.list_2_json(dev_examples))
        if self.training_args.do_predict:
            test_examples = self.get_examples("test")
            raw_datasets["test"] = DatasetK.from_dict(self.list_2_json(test_examples))

        if self.post_tokenizer:
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets
            return raw_datasets


        for key, value in raw_datasets.items():
            value.set_cache_files(["cache_local"])
        # remove_columns = self.sentence1_key if not self.sentence2_key else [
        #     self.sentence1_key, self.sentence2_key
        # ]
        tokenize_func = self.build_preprocess_function()
        load_from_cache_file = not self.data_args.overwrite_cache if self.training_args.local_rank in [-1, 0] else True
        base_cache_dir = os.path.join(self.model_args.cache_dir, "instruction_prompting") if self.model_args.cache_dir else os.path.join(os.path.expanduser("~"), ".cache/huggingface/datasets/")
        cache_dir = os.path.join(base_cache_dir, self.data_args.task_name, self.data_name)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        with self.training_args.main_process_first(desc="dataset tokenizer map"):
            raw_datasets = raw_datasets.map(
                tokenize_func,
                batched=True,
                load_from_cache_file=load_from_cache_file,
                num_proc=self.data_args.preprocessing_num_workers,
                desc="Running tokenizer on dataset",
                cache_file_names={k: f"{cache_dir}/cache_{self.data_args.task_name}_{self.data_name}_{str(k)}.arrow" for k in raw_datasets},
                # remove_columns=remove_columns
            )
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets

            return raw_datasets


    def build_preprocess_function(self):
        # Tokenize the texts
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length
        if self.model_args.model_type in ["gpt2", "opt", "gpt-neo", "llama"]:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        def func(examples):
            
            tokenized_examples = dict()
            # Tokenize
            tokenized_chosen_examples = tokenizer(
                examples[self.sentence1_key],
                text_pair=None,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length"
                if self.data_args.pad_to_max_length else False,
                # return_offsets_mapping=True
            )

            tokenized_examples["chosen_sequences"] = tokenized_chosen_examples["input_ids"]
            tokenized_examples["chosen_attention_mask"] = tokenized_chosen_examples["attention_mask"]

            tokenized_rejected_examples = tokenizer(
                examples[self.sentence2_key],
                text_pair=None,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length"
                if self.data_args.pad_to_max_length else False,
                # return_offsets_mapping=True
            )

            tokenized_examples["rejected_sequences"] = tokenized_rejected_examples["input_ids"]
            tokenized_examples["rejected_attention_mask"] = tokenized_rejected_examples["attention_mask"]

            return tokenized_examples

        return func
