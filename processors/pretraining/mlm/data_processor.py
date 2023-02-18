# -*- coding: utf-8 -*-
# @Time    : 2021/12/2 4:49 下午
# @Author  : JianingWang
# @File    : MLMProcessor
import logging
from itertools import chain
from typing import List

from datasets import load_dataset
from transformers import EvalPrediction
from processors.ProcessorBase import DataProcessor
from datasets import Dataset, load_from_disk
from processors.pretraining.mlm.data_collator import DataCollatorForMaskedLMWithoutNumber, DataCollatorForMaskedLM
from tools.processing_utils.common import is_chinese_char, is_chinese

logger = logging.getLogger(__name__)


# class MLMFromDisk(DataProcessor):
#     def __init__(self, data_args, training_args, model_args):
#         super().__init__(data_args, training_args, model_args)

#     def get_data_collator(self):
#         pad_to_multiple_of_8 = self.data_args.line_by_line and self.training_args.fp16 and not self.data_args.pad_to_max_length
#         return DataCollatorForMaskedLMWithoutNumber(
#             tokenizer=self.tokenizer,
#             mlm_probability=self.data_args.mlm_probability,
#             pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
#         )

#     def get_tokenized_datasets(self):
#         return load_from_disk(self.data_args.data_dir)

#     def compute_metrics(self, p: EvalPrediction):
#         preds = p.predictions[p.label_ids != -100]
#         labels = p.label_ids[p.label_ids != -100]
#         acc = (preds == labels).mean()
#         return {
#             'acc': round(acc, 4)
#         }

# class WWMFromDisk(MLMFromDisk):
#     def __init__(self, data_args, training_args, model_args):
#         super().__init__(data_args, training_args, model_args)

#     def get_data_collator(self):
#         pad_to_multiple_of_8 = self.data_args.line_by_line and self.training_args.fp16 and not self.data_args.pad_to_max_length
#         return DataCollatorForWholeWordMask(
#             tokenizer=self.tokenizer,
#             mlm_probability=self.data_args.mlm_probability,
#             pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
#         )

"""
Processing data for Masked LM
The pre-training corpus is saved in 'txt' file. Each line is a sentence.
"""
class MLMTextLineProcessor(DataProcessor):
    def __init__(self, data_args, training_args, model_args):
        super().__init__(data_args, training_args, model_args)

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.data_args.line_by_line and self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForMaskedLM(
            tokenizer=self.tokenizer,
            mlm_probability=self.data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )
        # Data Collator Option:
        # return DataCollatorForMaskedLMWithoutNumber(
        #     tokenizer=self.tokenizer,
        #     mlm_probability=self.data_args.mlm_probability,
        #     pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        # )
        

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
        padding = "max_length" if self.data_args.pad_to_max_length else False

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
                desc="Running tokenizer on dataset",
            )
        return tokenized_datasets


# class MLMGroupProcessor(MLMLineByLineProcessor):
#     def __init__(self, data_args, training_args, model_args):
#         super().__init__(data_args, training_args, model_args)

#     def get_tokenized_datasets(self):
#         # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
#         # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
#         # efficient when it receives the `special_tokens_mask`.
#         raw_datasets = self.get_examples('None')

#         # First we tokenize all the texts.
#         if self.training_args.do_train:
#             column_names = raw_datasets["train"].column_names
#         else:
#             column_names = raw_datasets["validation"].column_names
#         text_column_name = "text" if "text" in column_names else column_names[0]
#         max_seq_length = self.tokenizer.model_max_length if self.data_args.max_seq_length is None else self.data_args.max_seq_length

#         tokenizer = self.tokenizer

#         def tokenize_function(examples):
#             return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

#         with self.training_args.main_process_first(desc="dataset map tokenization"):
#             tokenized_datasets = raw_datasets.map(
#                 tokenize_function,
#                 batched=True,
#                 num_proc=self.data_args.preprocessing_num_workers,
#                 remove_columns=column_names,
#                 load_from_cache_file=not self.data_args.overwrite_cache,
#                 desc="Running tokenizer on every text in dataset",
#             )

#         # Main data processing function that will concatenate all texts from our dataset and generate chunks of
#         # max_seq_length.
#         def group_texts(examples):
#             # Concatenate all texts.
#             concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#             total_length = len(concatenated_examples[list(examples.keys())[0]])
#             # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#             # customize this part to your needs.
#             if total_length >= max_seq_length:
#                 total_length = (total_length // max_seq_length) * max_seq_length
#             # Split by chunks of max_len.
#             result = {
#                 k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
#                 for k, t in concatenated_examples.items()
#             }
#             return result

#         # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
#         # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
#         # might be slower to preprocess.
#         #
#         # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
#         # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

#         with self.training_args.main_process_first(desc="grouping texts together"):
#             raw_datasets = tokenized_datasets.map(
#                 group_texts,
#                 batched=True,
#                 num_proc=self.data_args.preprocessing_num_workers,
#                 load_from_cache_file=not self.data_args.overwrite_cache,
#                 desc=f"Grouping texts in chunks of {max_seq_length}",
#             )

#         return raw_datasets

#     def compute_metrics(self, p: EvalPrediction):
#         preds = p.predictions[p.label_ids != -100]
#         labels = p.label_ids[p.label_ids != -100]
#         acc = (preds == labels).mean()
#         return {
#             'eval_acc': round(acc, 4)
#         }


# def get_chinese_word(tokens: List[str]):
#     word_set = set()

#     for token in tokens:
#         chinese_word = len(token) > 1 and is_chinese(token)
#         if chinese_word:
#             word_set.add(token)
#     word_list = list(word_set)
#     return word_list


# def add_sub_symbol(bert_tokens: List[str], chinese_word_set):
#     if not chinese_word_set:
#         return bert_tokens
#     max_word_len = max([len(w) for w in chinese_word_set])

#     bert_word = bert_tokens
#     start, end = 0, len(bert_word)
#     while start < end:
#         single_word = True
#         if is_chinese(bert_word[start]):
#             l = min(end - start, max_word_len)
#             for i in range(l, 1, -1):
#                 whole_word = "".join(bert_word[start: start + i])
#                 if whole_word in chinese_word_set:
#                     for j in range(start + 1, start + i):
#                         bert_word[j] = "##" + bert_word[j]
#                     start = start + i
#                     single_word = False
#                     break
#         if single_word:
#             start += 1
#     return bert_word


# class WWMGroupProcessor(MLMLineByLineProcessor):
#     def __init__(self, data_args, training_args, model_args):
#         super().__init__(data_args, training_args, model_args)

#     def get_data_collator(self):
#         pad_to_multiple_of_8 = self.data_args.line_by_line and self.training_args.fp16 and not self.data_args.pad_to_max_length
#         return DataCollatorForWholeWordMask(
#             tokenizer=self.tokenizer,
#             mlm_probability=self.data_args.mlm_probability,
#             pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
#         )

#     def get_tokenized_datasets(self):
#         # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
#         # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
#         # efficient when it receives the `special_tokens_mask`.
#         raw_datasets = self.get_examples('None')

#         # First we tokenize all the texts.
#         if self.training_args.do_train:
#             column_names = raw_datasets["train"].column_names
#         else:
#             column_names = raw_datasets["validation"].column_names
#         text_column_name = "text" if "text" in column_names else column_names[0]
#         max_seq_length = self.tokenizer.model_max_length if self.data_args.max_seq_length is None else self.data_args.max_seq_length

#         tokenizer = self.tokenizer
#         import jieba

#         def tokenize_function(examples):
#             examples[text_column_name] = [
#                 line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
#             ]
#             bert_token = tokenizer(examples[text_column_name], return_special_tokens_mask=True)
#             word_token = [list(jieba.cut(i)) for i in examples[text_column_name]]
#             word_token = [get_chinese_word(w) for w in word_token]

#             ref_ids = []
#             for input_ids, chinese_word in zip(bert_token['input_ids'], word_token):
#                 input_tokens = []
#                 for id in input_ids:
#                     token = tokenizer._convert_id_to_token(id)
#                     input_tokens.append(token)
#                 input_tokens = add_sub_symbol(input_tokens, chinese_word)
#                 ref_id = [0] * len(input_ids)
#                 # We only save pos of chinese subwords start with ##, which mean is part of a whole word.
#                 for i, token in enumerate(input_tokens):
#                     if token[:2] == "##":
#                         clean_token = token[2:]
#                         # save chinese tokens' pos
#                         if len(clean_token) == 1 and is_chinese_char(ord(clean_token)):
#                             ref_id[i] = 1
#                 ref_ids.append(ref_id)
#             bert_token['chinese_ref'] = ref_ids
#             return bert_token

#         with self.training_args.main_process_first(desc="dataset map tokenization"):
#             tokenized_datasets = raw_datasets.map(
#                 tokenize_function,
#                 batched=True,
#                 num_proc=self.data_args.preprocessing_num_workers,
#                 remove_columns=column_names,
#                 load_from_cache_file=not self.data_args.overwrite_cache,
#                 desc="Running tokenizer on every text in dataset",
#             )

#         # Main data processing function that will concatenate all texts from our dataset and generate chunks of
#         # max_seq_length.
#         def group_texts(examples):
#             # Concatenate all texts.
#             concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#             total_length = len(concatenated_examples[list(examples.keys())[0]])
#             # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#             # customize this part to your needs.
#             if total_length >= max_seq_length:
#                 total_length = (total_length // max_seq_length) * max_seq_length
#             # Split by chunks of max_len.
#             result = {
#                 k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
#                 for k, t in concatenated_examples.items()
#             }
#             ref = []
#             for t in result['chinese_ref']:
#                 ref.append([i for i, j in enumerate(t) if j == 1])
#             result['chinese_ref'] = ref
#             return result

#         with self.training_args.main_process_first(desc="grouping texts together"):
#             raw_datasets = tokenized_datasets.map(
#                 group_texts,
#                 batched=True,
#                 num_proc=self.data_args.preprocessing_num_workers,
#                 load_from_cache_file=not self.data_args.overwrite_cache,
#                 desc=f"Grouping texts in chunks of {max_seq_length}",
#             )

#         return raw_datasets

#     def compute_metrics(self, p: EvalPrediction):
#         preds = p.predictions[p.label_ids != -100]
#         labels = p.label_ids[p.label_ids != -100]
#         acc = (preds == labels).mean()
#         return {
#             'acc': round(acc, 4)
#         }
