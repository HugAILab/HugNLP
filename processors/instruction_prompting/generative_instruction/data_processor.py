# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 9:13 下午
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
from processors.instruction_prompting.generative_instruction.data_collator import DataCollatorForCausalLM
from processors.basic_processors.prompt_processor import PromptBaseProcessor
from tools.processing_utils.tokenizer.tokenizer_utils import get_special_token_mapping


"""
The data processor for the generative instruction-tuning.
"""
class GenerativeInstructionProcessor(CLSProcessor):
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

        self.data_dir = data_args.data_dir
        self.train_file = os.path.join(data_args.data_dir, "train.json")
        self.dev_file = os.path.join(data_args.data_dir, "dev.json")
        self.test_file = os.path.join(data_args.data_dir, "test.json")
        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.text_key = "text"
        self.input_key = "input"
        self.output_key = "output"

    def get_data_collator(self):
        return DataCollatorForCausalLM(
            self.tokenizer,
            max_length=self.data_args.max_seq_length,
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
        """
        lines example:
        {
            "type": "text_only",
            "instances": [
                {
                    "text": "Input: Instruction: What is the scientific name for a beaver? \n Output: The scientific name for a beaver is Castor canadensis. \n\n"
                },
            ]
        }
        """
        lines = lines[0]
        examples = list()
        for ei, line in enumerate(lines["instances"]):
            idx = "{}-{}".format(set_type, str(ei))
            sentence = ""
            text = line[self.text_key] if self.text_key in line.keys() else ""
            input_text = line[self.input_key] if self.input_key in line.keys() else ""
            output_text = line[self.output_key] if self.output_key in line.keys() else ""
            sentence += text + input_text + output_text
            examples.append({
                "idx": idx,
                self.input_key: sentence,
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

        # remove_columns = [self.sentence_key]
        tokenize_func = self.build_preprocess_function()

        with self.training_args.main_process_first(desc="dataset tokenizer map"):
            raw_datasets = raw_datasets.map(
                tokenize_func,
                batched=True,
                desc="Running tokenizer on dataset",
                # remove_columns=remove_columns
            )
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets
            return raw_datasets

    def build_preprocess_function(self):
        # Tokenize the texts
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length

        def func(examples):

            # Tokenize
            tokenized_examples = tokenizer(
                examples[self.input_key],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length" if self.data_args.pad_to_max_length else False,
            )
            return tokenized_examples

        return func
