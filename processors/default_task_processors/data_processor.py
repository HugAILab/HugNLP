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
from processors.default_task_processors.data_collator import DataCollatorForDefaultSequenceClassification
from processors.basic_processors.prompt_processor import AddPromptIntoExample
from tools.processing_utils.tokenizer.tokenizer_utils import get_special_token_mapping


def sofmax(logits):
    probs = torch.softmax(torch.from_numpy(logits).float(), -1).numpy()
    return probs


"""
The data processor for the default sequence classification.
"""
class DefaultSequenceClassificationProcessor(CLSProcessor):
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
        self.data_dir = data_args.data_dir
        self.train_file = os.path.join(
            data_args.data_dir, "train.json"
        )  # each line: {"sentence1": xx, "sentence2": xx, "label": xx}
        self.dev_file = os.path.join(data_args.data_dir, "dev.json")
        self.test_file = os.path.join(data_args.data_dir, "test.json")
         # each line is one label name
        self.label_file = os.path.join(data_args.data_dir,"label_names.txt")
        self.template_file = os.path.join(data_args.data_dir, "template.json")
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


        if self.model_args.use_prompt_for_cls:
            # if use prompt, please first design a

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
            label_words_mapping = json.load(open(self.label_words_mapping_file, "r", encoding="utf-8"))


            self.prompt_engineering = AddPromptIntoExample(
                data_args=self.data_args,
                task_name=self.data_name,
                tokenizer=self.tokenizer,
                sentence1_key=self.sentence1_key,
                sentence2_key=self.sentence2_key,
                template=template,
                label_words_mapping=label_words_mapping)

            self.label_word_list = self.prompt_engineering.obtain_label_word_list()

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForDefaultSequenceClassification(
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
            sentence1 = line[self.sentence1_key]
            sentence2 = line[self.sentence2_key] if self.sentence2_key in line.keys() else None
            if set_type != "test":
                label = line["label"]
                if label not in self.labels:
                    continue
            else:
                # 有些测试集没有标签，为了避免报错，默认初始化标签0
                label = line["label"] if "label" in line.keys() else self.labels[0]

            label = self.label2id[label]

            examples.append({
                "idx": idx,
                self.sentence1_key: sentence1,
                self.sentence2_key: sentence2,
                "label": label
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

        remove_columns = self.sentence1_key if not self.sentence2_key else [
            self.sentence1_key, self.sentence2_key
        ]
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

            # adding prompt into each example
            if self.model_args.use_prompt_for_cls:
                # if use prompt, insert template into example
                examples = self.prompt_engineering.prompt_preprocess_function(examples)

            # Tokenize
            # print("examples["text_b"]=", examples["text_b"])
            if examples[self.sentence2_key][0] == None:
                text_pair = None
            else:
                text_pair = examples[self.sentence2_key]
            tokenized_examples = tokenizer(
                examples[self.sentence1_key],
                text_pair=text_pair,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length"
                if self.data_args.pad_to_max_length else False,
                # return_offsets_mapping=True
            )
            # 确定label
            if self.model_args.use_prompt_for_cls:
                mask_pos = []
                for input_ids in tokenized_examples["input_ids"]:
                    mask_pos.append(input_ids.index(get_special_token_mapping(self.tokenizer)["mask"]))
                tokenized_examples["mask_pos"] = mask_pos
            return tokenized_examples

        return func

    def get_predict_result(self, logits, examples, stage="dev"):
        if type(logits) == tuple:
            logits = logits[0]
        # logits: [test_data_num, label_num]
        predictions = dict() # 获取概率最大的作为预测结果
        topk_result = dict() # 根据概率取TopK个
        pseudo_data = list() # 根据预测的概率生成伪标签数据
        preds = logits
        preds = np.argmax(preds, axis=1)

        for pred, example, logit in zip(preds, examples, logits):
            id_ = example["idx"]
            id_ = int(id_.split("-")[1])
            predictions[id_] = pred  # 保存预测结果索引labelid
            # 获取TopK结果
            # {"prob": prob, "answer": answer}
            # print("logit=", logit)
            proba = sofmax(logit)  # 转换为概率
            # print("proba=", proba)
            # print("========")
            indices = np.argsort(-proba)  # 获得降序排列后的索引
            out = list()
            for index in indices[:20]:  # 依次取出相应的logit
                prob = proba[index].tolist()
                index = index.tolist()
                out.append({"prob": prob, "answer": index})
            topk_result[id_] = out

            pseudo_proba = proba[pred]
            # pseudo_predicts[id_] = {"label": pred, "pseudo_proba": pseudo_proba}

            # 顺便保存一下pseudo data
            # if pseudo_proba >= 0.99:
            pseudo_data.append({
                "idx": str(id_),
                self.sentence1_key: example[self.sentence1_key],
                self.sentence2_key: example[self.sentence2_key],
                "label": str(pred),
                "pseudo_proba": str(pseudo_proba)
            })

        # 保存标签结果
        with open(os.path.join(self.data_dir, "{}_pseudo.json".format(stage)),"w") as writer:
            for i, pred in enumerate(pseudo_data):
                json_d = pred
                writer.write(json.dumps(json_d, ensure_ascii=False) + "\n")

        return predictions, topk_result

    def compute_metrics(self, eval_predictions):
        examples = self.raw_datasets["validation"]
        labels = examples["label"]

        golden, dataname_map, dataname_type = {}, defaultdict(list), {}
        predictions, _ = self.get_predict_result(
            eval_predictions[0], examples, stage="dev")  # {"xx": "xxx", ...}
        for example in examples:
            data_type = "classification"
            data_name = self.data_name
            if data_name not in dataname_type:
                dataname_type[data_name] = data_type
            id_ = example["idx"]
            id_ = int(id_.split("-")[1])
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
        predicts, topk_predictions = self.get_predict_result(logits, examples, stage="test")
        label_list = self.labels
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
        topfile = os.path.join(self.training_args.output_dir, "top20_predict.json")
        with open(topfile, "w", encoding="utf-8") as f2:
            json.dump(topk_predictions, f2, ensure_ascii=False, indent=4)
