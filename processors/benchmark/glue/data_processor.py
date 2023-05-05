# -*- coding: utf-8 -*-
# @Time    : 2023/2/20 17:13 p.m.
# @Author  : JianingWang
# @File    : data_processor.py

import json
import torch
import random
import os.path
import numpy as np
from tqdm import tqdm
from datasets import DatasetDict
from datasets.load import load_dataset
from dataclasses import dataclass
from collections import defaultdict
from processors.dataset import DatasetK
from processors.ProcessorBase import CLSProcessor
from processors.benchmark.glue.glue_processor import glue_processors, task_to_keys, output_modes_mapping
from metrics import datatype2metrics
from tools.computations.softmax import softmax
from tools.model_utils.gpt_response import GPTResponse
from processors.benchmark.glue.data_collator import DataCollatorForGLUE, DataCollatorForGLUEInContextLearning
from processors.basic_processors.prompt_processor import PromptBaseProcessor, InstructionPromptProcessor
from tools.processing_utils.tokenizer.tokenizer_utils import get_special_token_mapping
from processors.benchmark.glue.task_engineering import masked_task_to_template, causal_task_to_template, task_to_instruction, label_words_mapping

from tools.runner_utils.log_util import logging

logger = logging.getLogger(__name__)

"""
Default GLUE
"""
class GLUEProcessor(CLSProcessor):
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
        assert "data_name" in param, "You must add one defined param 'data_name=xxx' in the user_defined parameter."
        self.data_name = param["data_name"]
        self.output_modes = output_modes_mapping[self.data_name]
        self.max_seq_length = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride

        self.raw_datasets = load_dataset("glue", self.data_name)
        self.labels = self.raw_datasets["train"].features["label"].names
        self.sentence1_key, self.sentence2_key = task_to_keys[self.data_name]

        if self.model_args.use_prompt_for_cls:
            # if use prompt, please first perform task prompt engineering in task_engineering.py

            self.prompt_engineering = PromptBaseProcessor(
                data_args=self.data_args,
                task_name=self.data_name,
                tokenizer=self.tokenizer,
                sentence1_key=self.sentence1_key,
                sentence2_key=self.sentence2_key,
                template=masked_task_to_template[self.data_name],
                label_words_mapping=label_words_mapping[self.data_name])

            if len(self.labels) > 1:
                self.label_word_list = self.prompt_engineering.obtain_label_word_list()
            else:
                # Regression task
                # "0" represents low polarity and "1" represents high polarity.
                self.label_word_list = [self.label_to_word[label] for label in ["0", "1"]]

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForGLUE(
            self.tokenizer,
            max_length=self.data_args.max_seq_length,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            pad_to_max_length=self.data_args.pad_to_max_length)

    def get_tokenized_datasets(self):

        raw_datasets = self.raw_datasets

        if self.post_tokenizer:
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets
            return raw_datasets
        # datasets的bug, 对于from_dict不会创建cache,需要指定cache_file_names
        # 指定了cache_file_names在_map_single中也需要cache_files不为空才能读取cache
        # for key, value in raw_datasets.items():
        #     value.set_cache_files(["cache_local"])
        remove_columns = self.sentence1_key if not self.sentence2_key else [
            self.sentence1_key, self.sentence2_key
        ]
        tokenize_func = self.build_preprocess_function()
        # 多gpu, 0计算完存cache，其他load cache
        load_from_cache_file = not self.data_args.overwrite_cache if self.training_args.local_rank in [-1, 0] else True
        base_cache_dir = os.path.join(self.model_args.cache_dir, "datasets") if self.model_args.cache_dir else os.path.join(os.path.expanduser("~"), ".cache/huggingface/datasets/")
        cache_dir = os.path.join(base_cache_dir, self.data_args.task_name, self.data_name)

        os.makedirs(cache_dir, exist_ok=True)
        with self.training_args.main_process_first(desc="dataset tokenizer map"):
            raw_datasets = raw_datasets.map(
                tokenize_func,
                batched=True,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
                # cache_file_names={k: f"{cache_dir}/cache_{self.data_args.task_name}_{self.data_name}_{str(k)}.arrow" for k in raw_datasets},
                # remove_columns=remove_columns
            )
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets

            return raw_datasets

    def get_examples(self, set_type):
        pass

    def _create_examples(self, lines, set_type):
        pass

    def build_preprocess_function(self):
        def func(examples):
            # add by wjn
            # adding prompt into each example
            if self.model_args.use_prompt_for_cls:
                # if use prompt, insert template into example
                examples = self.prompt_engineering.add_prompt_into_example(examples)

            # Tokenize the texts
            args = ((examples[self.sentence1_key], )if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key]))
            result = self.tokenizer(
                *args,
                padding="max_length"
                if self.data_args.pad_to_max_length else False,
                max_length=self.max_seq_length,
                truncation=True
            )

            if self.model_args.use_prompt_for_cls:
                mask_pos = []
                for input_ids in result["input_ids"]:
                    mask_pos.append(input_ids.index(get_special_token_mapping(self.tokenizer)["mask"]))
                result["mask_pos"] = mask_pos

            return result

        return func

    def get_predict_result(self, logits, examples, stage="dev"):
        # logits: [test_data_num, label_num]
        predictions = dict()  # 获取概率最大的作为预测结果
        topk_result = dict()  # 根据概率取TopK个
        pseudo_data = list()  # 根据预测的概率生成伪标签数据
        preds = logits
        if self.output_modes == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.output_modes == "regression":
            preds = np.squeeze(preds)

        for pred, example, logit in zip(preds, examples, logits):
            id_ = example["idx"]
            predictions[id_] = pred  # 保存预测结果索引labelid
            # 获取TopK结果
            # {"prob": prob, "answer": answer}
            # print("logit=", logit)
            proba = softmax(logit)  # 转换为概率
            # print("proba=", proba)
            # print("========")
            indices = np.argsort(-proba)  # 获得降序排列后的索引
            out = list()
            for index in indices[:20]:  # 依次取出相应的logit
                prob = proba[index].tolist()
                index = index.tolist()
                out.append({"prob": prob, "answer": index})
            topk_result[id_] = out

        return predictions, topk_result

    def compute_metrics(self, eval_predictions):
        examples = self.raw_datasets["validation"]
        labels = examples["label"]

        golden, dataname_map, dataname_type = {}, defaultdict(list), {}
        predictions, _ = self.get_predict_result(
            eval_predictions[0], examples, stage="dev")  # {"xx": "xxx", ...}
        for example in examples:
            data_type = self.output_modes
            data_name = self.data_name
            if data_name not in dataname_type:
                dataname_type[data_name] = data_type
            id_ = example["idx"]
            dataname_map[data_name].append(id_)
            golden[id_] = example["label"]

        all_metrics = {
            "eval_macro_f1": 0.,
            "eval_micro_f1": 0.,
            "eval_num": 0,
            "eval_acc": 0.,
        }

        for dataname, data_ids in dataname_map.items():
            metric = datatype2metrics[dataname_type[dataname]]()
            gold = {k: v for k, v in golden.items() if k in data_ids}
            pred = {k: v for k, v in predictions.items() if k in data_ids}
            # pred = {"dev-{}".format(value["id"]): value["label"] for value in predictions if "dev-{}".format(value["id"]) in data_ids}
            score = metric.calc_metric(golden=gold, predictions=pred)
            acc, f1 = score["acc"], score["f1"]
            if len(gold) != len(pred) or len(gold) < 20:
                # print(dataname, dataname_type[dataname], round(acc, 4), len(gold), len(pred), data_ids)
                print("len(gold)=", len(gold))
                print("len(pred)=", len(pred))
            all_metrics["eval_macro_f1"] += f1
            all_metrics["eval_micro_f1"] += f1 * len(data_ids)
            all_metrics["eval_num"] += len(data_ids)
            all_metrics["eval_acc"] += acc
            all_metrics[dataname] = round(f1, 4)
        all_metrics["eval_macro_f1"] = round(
            all_metrics["eval_macro_f1"] / len(dataname_map), 4)
        all_metrics["eval_micro_f1"] = round(
            all_metrics["eval_micro_f1"] / all_metrics["eval_num"], 4)
        all_metrics["eval_macro_acc"] = round(
            all_metrics["eval_acc"] / len(dataname_map), 4)

        return all_metrics

    def save_result(self, logits, label_ids):
        examples = self.raw_datasets["test"]
        predicts, topk_predictions = self.get_predict_result(logits,
                                                             examples,
                                                             stage="test")
        clue_processor = glue_processors[self.data_name]()
        label_list = clue_processor.get_labels()
        id2label = {i: label for i, label in enumerate(label_list)}

        ### submit 格式转换为clue的
        answer = list()
        for k, v in predicts.items():
            if v not in id2label.keys():
                res = ""
                # print("unknow answer: {}".format(v))
                print("unknown")
            else:
                res = id2label[v]
            answer.append({"id": k, "label": res})

        output_submit_file = os.path.join(self.training_args.output_dir, "answer.json")
        # 保存标签结果
        with open(output_submit_file, "w") as writer:
            for i, pred in enumerate(answer):
                json_d = {}
                json_d["id"] = i
                json_d["label"] = pred["label"]
                writer.write(json.dumps(json_d) + "\n")

        # 保存TopK个预测结果
        topfile = os.path.join(self.training_args.output_dir,
                               "top20_predict.json")
        with open(topfile, "w", encoding="utf-8") as f2:
            json.dump(topk_predictions, f2, ensure_ascii=False, indent=4)

"""
GLUE for causal cls.
e.g., leverage GPT-2 for in-context learning
"""
class GLUEForInContextProcessor(CLSProcessor):

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
        self.data_name = param["data_name"]
        self.output_modes = output_modes_mapping[self.data_name]
        self.max_seq_length = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride

        self.raw_datasets = load_dataset("glue", self.data_name)
        self.labels = self.raw_datasets["train"].features["label"].names
        self.sentence1_key, self.sentence2_key = task_to_keys[self.data_name]

        self.data_dir = self.data_args.data_dir
        self.num_incontext_example = int(param["num_incontext_example"]) if "num_incontext_example" in param.keys() else 0 # the number of in-context example
        self.l = int(param["l"]) if "l" in param.keys() else 1 # the max length to generate
        self.use_calibrate = param["use_calibrate"] == "True" if "use_calibrate" in param.keys() else False # whether to calibrate the prediction
        self.content_free = ["N/A"] # When set "use_calibrate" is True, will add a new content_free example.

        template = causal_task_to_template[self.data_name]
        instruction = task_to_instruction[self.data_name]
        # label_words_mapping: {"xxx": ["xxx"], ...}
        self.label_words_mapping = label_words_mapping[self.data_name]

        self.label2id = {label: ei for ei, label in enumerate(self.labels)}
        self.id2label = {ei: label for ei, label in enumerate(self.labels)}

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


    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForGLUEInContextLearning(self.tokenizer, pad_to_multiple_of=8 if pad_to_multiple_of_8 else None, pad_to_max_length=self.data_args.pad_to_max_length)

    def InContextSampling(self, examples: list) -> list:
        # used for sampling in-context examples
        random.shuffle(examples)
        incontext_examples = examples[:self.num_incontext_example] if self.num_incontext_example < len(examples) else examples
        return incontext_examples


    def get_tokenized_datasets(self):
        raw_datasets = self.raw_datasets

        raw_datasets["validation"] = DatasetK.from_dict(self.list_2_json(self.get_prompt_examples(raw_datasets["train"], raw_datasets["validation"], set_type="dev")))
        raw_datasets["test"] = DatasetK.from_dict(self.list_2_json(self.get_prompt_examples(raw_datasets["train"], raw_datasets["test"], set_type="test")))

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
        pass

    def get_prompt_examples(self, training_examples, eval_examples, set_type: str):
        # assert set_type != "train", "In-context learning dose not have training proce"

        # 随机采样若干in-context example作为demonstration
        if self.incontext_examples is None:
            training_example_list = list()
            for ei, example in enumerate(training_examples):
                training_example_list.append({
                    "idx": example["idx"],
                    self.sentence1_key: example[self.sentence1_key],
                    self.sentence2_key if self.sentence2_key is not None else "sentence2": example[self.sentence2_key] if self.sentence2_key in example.keys() else "",
                    "label": example["label"],
                    "target": self.id2label[example["label"]],
                })
            self.incontext_examples = self.InContextSampling(training_example_list)

        # 为每个dev/test构建prompt
        prompt_eval_examples = list()
        for ei, example in enumerate(tqdm(eval_examples)):
            prompt = self.prompt_engineering.construct_incontext_prompt(
                sentence1_key=self.sentence1_key,
                sentence2_key=self.sentence2_key,
                incontext_examples=self.incontext_examples,
                eval_example=example
                )
            if set_type != "test":
                prompt_eval_examples.append({
                    "idx": example["idx"],
                    self.sentence1_key: prompt,
                    self.sentence2_key if self.sentence2_key is not None else "sentence2": "",
                    "label": example["label"],
                    "target": self.id2label[example["label"]],
                })
            else:
                prompt_eval_examples.append({
                    "idx": ei,
                    self.sentence1_key: prompt,
                    self.sentence2_key if self.sentence2_key is not None else "sentence2": "",
                    "label": 0, # the label of test set is missing, so default initialize as 0
                    "target": self.id2label[0], # the label of test set is missing, so default initialize as 0
                })
        print("prompt_eval_examples[0]=", prompt_eval_examples[0])

        return prompt_eval_examples # List[dict]

    def _create_examples(self, lines, set_type=None):
        pass

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
