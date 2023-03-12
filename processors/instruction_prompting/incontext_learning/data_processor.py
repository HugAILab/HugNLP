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

from tools.model_utils.gpt_response import GPTResponse

"""
GPT for classification in-context learning
"""
class ClassificationInContextLearningProcessor(CLSProcessor):
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
        self.data_dir = data_args.data_dir
        self.train_file = os.path.join(
            data_args.data_dir, "train.json"
        )  # each line: {"sentence1": xx, "sentence2": xx, "label": xx}
        self.dev_file = os.path.join(data_args.data_dir, "dev.json")
        self.test_file = os.path.join(data_args.data_dir, "test.json")
        assert os.path.exists(self.test_file), "the test.json is not found in {}".format(self.data_dir)
         # each line is one label name
        self.label_file = os.path.join(data_args.data_dir,"label_names.txt")
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
        elif os.path.exists(self.label_file):
            self.labels = list()
            with open(self.label_file, "r", encoding="utf-8") as fr:
                lines = fr.readlines()
            for line in lines:
                self.labels.append(line.replace("\n", ""))
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
            self.api_response = GPTResponse("gpt2") # default for gpt2

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token # GPT需要显式地添加padding token

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForClassificationInContextLearning(self.tokenizer, pad_to_multiple_of=8 if pad_to_multiple_of_8 else None, pad_to_max_length=self.data_args.pad_to_max_length)

    def InContextSampling(self, examples: list) -> list:
        # used for sampling in-context examples
        random.seed = self.data_args.seed
        random.shuffle(examples)
        incontext_examples = examples[:self.num_incontext_example] if self.num_incontext_example > len(examples) else examples
        return incontext_examples

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
            incontext_examples = self.InContextSampling(training_examples)
            # incontext_examples = self.incontext_examples

            examples = self._create_examples(self._read_json2(self.dev_file), set_type)
            # 为每个dev/test构建prompt
            eval_examples = list()
            for example in examples:
                prompt = self.prompt_engineering.construct_incontext_prompt(incontext_examples, example)
                eval_examples.append({
                    "idx": example["idx"],
                    self.sentence1_key: prompt,
                    self.sentence2_key: "",
                    "label": example["label"],
                    "target": example["target"],
                })

        return examples # List[dict]

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
                max_length=max_seq_length,
                padding="max_length" if self.data_args.pad_to_max_length else False,
                # return_offsets_mapping=True
            )
            # 确定label
            return tokenized_examples

        return func


    def get_predict_result(self, logits, examples):
        probs, indices = logits

        self.api_response.call_for_gpt2_response(
            gpt2_tokenizer=self.tokenizer,
            logits=probs,
        )

        probs = probs.squeeze(1)  # topk结果的概率
        indices = indices.squeeze(1)  # topk结果的索引
        # print("probs=", probs) # [n, m]
        # print("indices=", indices) # [n, m]
        predictions = {}
        topk_predictions = {}
        for prob, index, example in zip(probs, indices, examples):
            id_ = example["idx"]
            index_ids = torch.Tensor([i for i in range(len(index))]).long()
            topk_answer = list()

            best_start_end = np.unravel_index(index[0], (self.data_args.max_seq_length, self.data_args.max_seq_length))
            s = example["offset_mapping"][best_start_end[0]][0]
            e = example["offset_mapping"][best_start_end[1]][1]
            answer = example["content"][s: e]
            predictions[id_] = answer

            topk_answer_dict = dict()
            topk_index = index[prob > 0.0]
            index_ids = index_ids[prob > 0.0]
            # print("index_ids=", index_ids)
            for ei, index in enumerate(topk_index):
                if ei > 6:
                    break
                # 1D index转2D index
                start_end = np.unravel_index(index, (self.data_args.max_seq_length, self.data_args.max_seq_length))
                s = example["offset_mapping"][start_end[0]][0]
                e = example["offset_mapping"][start_end[1]][1]
                ans = example["content"][s: e]
                # topk_answer.append({"answer": ans, "prob": float(prob[index_ids[ei]]), "pos": (s, e)})
                topk_answer_dict[ans] = {"prob": float(prob[index_ids[ei]]), "pos": [(s, e)]}

            predictions[id_] = answer
            if id_ not in topk_predictions.keys():
                topk_predictions[id_] = topk_answer_dict

        for id_, values in topk_predictions.items():
            # values {"ans": {}, ...}
            answer_list = list()
            for ans, value in values.items():
                answer_list.append({"answer": ans, "prob": value["prob"], "pos": value["pos"]})
            topk_predictions[id_] = answer_list

        return predictions, topk_predictions

    def compute_metrics(self, eval_predictions):
        examples = self.raw_datasets["validation"]
        golden, dataname_map, dataname_type = {}, defaultdict(list), {}
        predictions, _ = self.get_predict_result(eval_predictions[0], examples)
        for example in examples:
            data_type = example["data_type"]
            dataname = "_".join(example["id"].split("_")[:-1])
            if dataname not in dataname_type:
                dataname_type[dataname] = data_type
            id_ = example["id"]
            dataname_map[dataname].append(id_)
            if data_type == "ner":
                golden[id_] = example["target"].split("|")
            else:
                golden[id_] = example["target"]

        all_metrics = {
            "macro_f1": 0.,
            "micro_f1": 0.,
            "eval_num": 0,
        }

        for dataname, data_ids in dataname_map.items():
            metric = datatype2metrics[dataname_type[dataname]]()
            gold = {k: v for k, v in golden.items() if k in data_ids}
            pred = {k: v for k, v in predictions.items() if k in data_ids}
            score = metric.calc_metric(golden=gold, predictions=pred)
            # print("score=", score)
            acc, f1 = score["acc"], score["f1"]
            # if len(gold) != len(pred) or len(gold) < 20:
                # print(dataname, dataname_type[dataname], round(acc, 4), len(gold), len(pred), data_ids)
            all_metrics["macro_f1"] += f1
            all_metrics["micro_f1"] += f1 * len(data_ids)
            all_metrics["eval_num"] += len(data_ids)
            all_metrics[dataname] = round(acc, 4)
        all_metrics["macro_f1"] = round(all_metrics["macro_f1"] / len(dataname_map), 4)
        all_metrics["micro_f1"] = round(all_metrics["micro_f1"] / all_metrics["eval_num"], 4)
        return all_metrics

    def save_result(self, logits, label_ids):
        examples = self.raw_datasets["test"]
        predicts, topk_predicts = self.get_predict_result(logits, examples)
        # print("topk_predicts=", topk_predicts)

        outfile = os.path.join(self.training_args.output_dir, "answer.json")
        with open(outfile, "w", encoding="utf8") as f:
            json.dump(predicts, f, ensure_ascii=False, indent=2)

        topk_file = os.path.join(self.training_args.output_dir, "topk_prob.json")
        with open(topk_file, "w", encoding="utf8") as f2:
            json.dump(topk_predicts, f2, ensure_ascii=False, indent=2)
