# -*- coding: utf-8 -*-
# @Time    : 2021/12/2 4:49 下午
# @Author  : JianingWang
# @File    : causal_lm.py
import logging
from dataclasses import dataclass
from datasets import load_dataset
from transformers import EvalPrediction
from processors.ProcessorBase import DataProcessor
from processors.pretraining.causal_lm.data_collator import DataCollatorForCausalLM

logger = logging.getLogger(__name__)


"""
Processing data for Causal LM
The pre-training corpus is saved in 'txt' file. Each line is a sentence.
"""
class CausalLMITextLineProcessor(DataProcessor):
    def __init__(self, data_args, training_args, model_args, tokenizer=None):
        super().__init__(data_args, training_args, model_args)
        self.tokenizer = tokenizer

    def get_data_collator(self):
        return DataCollatorForCausalLM(
            tokenizer=self.tokenizer,
            pad_to_max_length=self.data_args.pad_to_max_length,
        )

    def get_examples(self, set_type=None):
        data_files = {}
        if self.data_args.train_file is not None:
            data_files["train"] = self.data_args.train_file
            extension = self.data_args.train_file.split(".")[-1]
        if self.data_args.validation_file is not None:
            data_files["validation"] = self.data_args.validation_file
            extension = self.data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=self.model_args.cache_dir)
        # raw_datasets['train'] = raw_datasets['train'].shuffle()
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{self.data_args.validation_split_percentage}%]",
                cache_dir=self.model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{self.data_args.validation_split_percentage}%:]",
                cache_dir=self.model_args.cache_dir,
            )
        return raw_datasets

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[p.label_ids != -100]
        labels = p.label_ids[p.label_ids != -100]
        acc = (preds == labels).mean()
        return {
            'eval_acc': round(acc, 4)
        }

    def get_tokenized_datasets(self):

        data_files = {}
        if self.data_args.train_file is not None:
            data_files["train"] = self.data_args.train_file
            extension = self.data_args.train_file.split(".")[-1]
        if self.data_args.validation_file is not None:
            data_files["validation"] = self.data_args.validation_file
            extension = self.data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=self.model_args.cache_dir)
        # raw_datasets['train'] = raw_datasets['train'].shuffle()
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{self.data_args.validation_split_percentage}%]",
                cache_dir=self.model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{self.data_args.validation_split_percentage}%:]",
                cache_dir=self.model_args.cache_dir,
            )
        logger.info(f'validation fingerprint {raw_datasets}')
        if self.training_args.do_train:
            column_names = raw_datasets["train"].column_names
        else:
            column_names = raw_datasets["validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        max_seq_length = self.tokenizer.model_max_length if self.data_args.max_seq_length is None else self.data_args.max_seq_length
        # When using line_by_line, we just tokenize each nonempty line.
        # padding = "max_length" if self.data_args.pad_to_max_length else False
        padding = False

        tokenizer = self.tokenizer

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            # examples['length'] = [len(line) for line in examples[text_column_name]]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )

        with self.training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
        return tokenized_datasets



# # 按照in-context format进行预训练
# class CausalLMInContextProcessor(DataProcessor):
#     def __init__(self, data_args, training_args, model_args):
#         super().__init__(data_args, training_args, model_args)

#     def get_data_collator(self):
#         return DataCollatorForInContextCausalLM(
#             tokenizer=self.tokenizer,
#             pad_to_max_length=self.data_args.pad_to_max_length,
#         )

#     def get_examples(self, set_type=None):
#         data_files = {}
#         if self.data_args.train_file is not None:
#             data_files["train"] = self.data_args.train_file
#             extension = self.data_args.train_file.split(".")[-1]
#         if self.data_args.validation_file is not None:
#             data_files["validation"] = self.data_args.validation_file
#             extension = self.data_args.validation_file.split(".")[-1]
#         if extension == "txt":
#             extension = "text"
#         raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=self.model_args.cache_dir)
#         # raw_datasets['train'] = raw_datasets['train'].shuffle()
#         # If no validation data is there, validation_split_percentage will be used to divide the dataset.
#         if "validation" not in raw_datasets.keys():
#             raw_datasets["validation"] = load_dataset(
#                 extension,
#                 data_files=data_files,
#                 split=f"train[:{self.data_args.validation_split_percentage}%]",
#                 cache_dir=self.model_args.cache_dir,
#             )
#             raw_datasets["train"] = load_dataset(
#                 extension,
#                 data_files=data_files,
#                 split=f"train[{self.data_args.validation_split_percentage}%:]",
#                 cache_dir=self.model_args.cache_dir,
#             )
#         return raw_datasets

#     def compute_metrics(self, p: EvalPrediction):
#         preds = p.predictions[p.label_ids != -100]
#         labels = p.label_ids[p.label_ids != -100]
#         acc = (preds == labels).mean()
#         return {
#             'eval_acc': round(acc, 4)
#         }

#     def get_tokenized_datasets(self):

#         data_files = {}
#         if self.data_args.train_file is not None:
#             data_files["train"] = self.data_args.train_file
#             extension = self.data_args.train_file.split(".")[-1]
#         if self.data_args.validation_file is not None:
#             data_files["validation"] = self.data_args.validation_file
#             extension = self.data_args.validation_file.split(".")[-1]
#         if extension == "json":
#             extension = "json"
#         raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=self.model_args.cache_dir)
#         # raw_datasets['train'] = raw_datasets['train'].shuffle()
#         # If no validation data is there, validation_split_percentage will be used to divide the dataset.
#         if "validation" not in raw_datasets.keys():
#             raw_datasets["validation"] = load_dataset(
#                 extension,
#                 data_files=data_files,
#                 split=f"train[:{self.data_args.validation_split_percentage}%]",
#                 cache_dir=self.model_args.cache_dir,
#             )
#             raw_datasets["train"] = load_dataset(
#                 extension,
#                 data_files=data_files,
#                 split=f"train[{self.data_args.validation_split_percentage}%:]",
#                 cache_dir=self.model_args.cache_dir,
#             )
#         logger.info(f'validation fingerprint {raw_datasets}')
#         if self.training_args.do_train:
#             column_names = raw_datasets["train"].column_names
#         else:
#             column_names = raw_datasets["validation"].column_names
#         text_column_name = "text" if "text" in column_names else column_names[0]
#         max_seq_length = self.tokenizer.model_max_length if self.data_args.max_seq_length is None else self.data_args.max_seq_length
#         # When using line_by_line, we just tokenize each nonempty line.
#         padding = "max_length" if self.data_args.pad_to_max_length else False

#         # tokenizer = self.tokenizer

#         # def tokenize_function(examples):
#         #     # Remove empty lines
#         #     examples[text_column_name] = [
#         #         line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
#         #     ]
#         #     # examples['length'] = [len(line) for line in examples[text_column_name]]
#         #     return tokenizer(
#         #         examples[text_column_name],
#         #         padding=padding,
#         #         truncation=True,
#         #         max_length=max_seq_length,
#         #         return_special_tokens_mask=True,
#         #     )

#         # with self.training_args.main_process_first(desc="dataset map tokenization"):
#         #     tokenized_datasets = raw_datasets.map(
#         #         tokenize_function,
#         #         batched=True,
#         #         num_proc=self.data_args.preprocessing_num_workers,
#         #         remove_columns=[text_column_name],
#         #         load_from_cache_file=not self.data_args.overwrite_cache,
#         #         desc="Running tokenizer on dataset line_by_line",
#         #     )
#         return raw_datasets
