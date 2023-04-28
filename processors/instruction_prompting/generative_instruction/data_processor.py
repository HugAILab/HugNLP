# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 9:13 下午
# @Author  : JianingWang
# @File    : data_processor.py
import json
import torch
import os.path
import numpy as np
from tqdm import tqdm
from itertools import chain
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

from tools.runner_utils.log_util import logging
logger = logging.getLogger(__name__)

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
        param = {p.split("=")[0]: p.split("=")[1] for p in (data_args.user_defined).split(" ")}
        self.causal_lm_name = "gpt2" if "causal_lm_name" not in param.keys() else param["causal_lm_name"] # e.g. GPT, OPT, ...
        self.stop_token = None if "stop_token" not in param.keys() else param["stop_token"] # e.g., <|endoftext|>
        self.language = "" if "language" not in param.keys() else param["language"] # e.g., en, zh
        self.data_dir = data_args.data_dir
        self.train_file = os.path.join(data_args.data_dir, "instruction_{}_corpora.json".format(self.language))
        self.dev_file = os.path.join(data_args.data_dir, "instruction_dev.json")
        self.test_file = os.path.join(data_args.data_dir, "instruction_test.json")
        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.data_name = "instruction_corpora"
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
        [
            {
                "text": "[Human]: Instruction: What is the scientific name for a beaver? \n [HugChat]: The scientific name for a beaver is Castor canadensis. \n\n"
            },
            ...
        ]

        """
        examples = list()
        for ei, line in enumerate(tqdm(lines)):
            idx = "{}-{}".format(set_type, str(ei))
            input_text = ""
            if set_type == "train":
                input_text = line[self.text_key] if self.text_key in line.keys() else ""
                if self.stop_token is not None:
                    input_text = input_text.strip() + " " + self.stop_token
                examples.append({
                    "idx": idx,
                    self.input_key: input_text,
                })
            else:
                input_text = line[self.input_key] if self.input_key in line.keys() else ""
                if self.stop_token is not None:
                    input_text = input_text.strip() + " " + self.stop_token
                output_text = line[self.output_key] if self.output_key in line.keys() else ""
                examples.append({
                    "idx": idx,
                    self.input_key: input_text,
                    self.output_key: output_text,
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
        # remove_columns = [self.sentence_key]
        tokenize_func = self.build_preprocess_function()
        load_from_cache_file = not self.data_args.overwrite_cache if self.training_args.local_rank in [-1, 0] else True
        base_cache_dir = os.path.join(self.model_args.cache_dir, "instruction_prompting") if self.model_args.cache_dir else os.path.join(os.path.expanduser("~"), ".cache/huggingface/datasets/")
        cache_dir = os.path.join(base_cache_dir, self.data_args.task_name, self.data_name)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        with self.training_args.main_process_first(desc="dataset tokenizer map"):
            tokenize_batch_size = 1000
            raw_datasets = raw_datasets.map(
                tokenize_func,
                batched=True,
                load_from_cache_file=True,
                batch_size=tokenize_batch_size,
                num_proc=self.data_args.preprocessing_num_workers,
                desc="Running tokenizer on dataset",
                cache_file_names={k: f"{cache_dir}/cache_{self.data_args.task_name}_{self.data_name}_{self.language}_{self.causal_lm_name}_{str(k)}.arrow" for k in raw_datasets},
                # remove_columns=remove_columns
            )
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets

            # raw_datasets = self.group_text(
            #     tokenized_datasets=raw_datasets,
            #     model_max_length=self.data_args.max_seq_length
            # )
            return raw_datasets


    def group_text(self, tokenized_datasets: DatasetDict, model_max_length):
        """
        Groups texts together to form blocks of maximum length `model_max_length` and returns the processed data as
        a dictionary.
        """
        data_args = self.data_args

        if data_args.block_size is None:
            block_size = model_max_length
            if block_size > 1024:
                logger.warning(
	    			"The chosen tokenizer supports a `model_max_length` that is"
	    			" longer than the default `block_size` value"
	    			" of 1024. If you would like to use a longer `block_size`"
	    			" up to `tokenizer.model_max_length` you can override this "
	    			" default with `--block_size xxx`."
                )
                block_size = 1024
        else:
            if data_args.block_size > model_max_length:
                logger.warning(
                    f"The block_size passed ({data_args.block_size}) is larger"
	    			f" than the maximum length for the model"
                    f"({model_max_length})."
                    f" Using block_size={model_max_length}."
                )
            block_size = min(data_args.block_size, model_max_length)

        # Main data processing function that will concatenate all texts from
        # our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model
            # supported it instead of this drop, you can customize this part to
            # your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts
        # together, so group_texts throws away a remainder for each of those
        # groups of 1,000 texts. You can adjust that batch_size here but a
        # higher value might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation
        # of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        load_from_cache_file = not self.data_args.overwrite_cache if self.training_args.local_rank in [-1, 0] else True
        base_cache_dir = os.path.join(self.model_args.cache_dir, "instruction_prompting") if self.model_args.cache_dir else os.path.join(os.path.expanduser("~"), ".cache/huggingface/datasets/")
        cache_dir = os.path.join(base_cache_dir, self.data_args.task_name, self.data_name + "_group")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        with self.training_args.main_process_first(desc="grouping texts together"):
            group_batch_size = 1000
            # if data_args.disable_group_texts:
            #     group_batch_size = 1
            tokenized_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                load_from_cache_file=load_from_cache_file,
                batch_size=group_batch_size,
                num_proc=data_args.preprocessing_num_workers,
                desc=f"Grouping texts in chunks of {block_size}",
                cache_file_names={k: f"{cache_dir}/cache_{self.data_args.task_name}_{self.data_name}_{str(k)}.arrow" for k in tokenized_datasets},
            )

        return tokenized_datasets



    def build_preprocess_function(self):
        # Tokenize the texts
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length

        def func(examples):

            # Tokenizes
            tokenized_examples = tokenizer(
                examples[self.input_key],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length" if self.data_args.pad_to_max_length else False,
            )
            tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
            return tokenized_examples

        return func
