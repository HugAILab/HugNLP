# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 9:13 下午
# @Author  : JianingWang
# @File    : clue
import json
import torch
import os.path
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from processors.ProcessorBase import CLSProcessor
from processors.benchmark.clue.clue_processor import clue_processors, clue_output_modes
from metrics import datatype2metrics
from tools.computations.softmax import softmax
from processors.benchmark.clue.data_collator import DataCollator


class CLUEProcessor(CLSProcessor):
    def __init__(self, data_args, training_args, model_args, tokenizer=None, post_tokenizer=False, keep_raw_data=True):
        super().__init__(data_args, training_args, model_args, tokenizer, post_tokenizer=post_tokenizer, keep_raw_data=keep_raw_data)
        param = {p.split("=")[0]: p.split("=")[1] for p in (data_args.user_defined).split(" ")}
        assert "data_name" in param, "You must add one defined param 'data_name=xxx' in the user_defined parameter."
        self.data_name = param["data_name"]
        self.is_pseudo = False # 是否加载上一轮模型预测的dev和test的pseudo label
        self.pseudo_threshold = 1.0
        if "is_pseudo" in param.keys():
            self.is_pseudo = bool(param["is_pseudo"])
            self.pseudo_threshold = float(param["pseudo_threshold"])
        self.data_dir = data_args.data_dir
        assert self.data_name in clue_processors.keys(), "Unknown task name {}".format(self.data_name)
        self.processor = clue_processors[self.data_name]()
        self.output_modes = clue_output_modes[self.data_name]
        self.train_file = os.path.join(data_args.data_dir, "train.json")
        self.dev_file = os.path.join(data_args.data_dir, "dev.json")
        self.test_file = os.path.join(data_args.data_dir, "test.json")
        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.sentence1_key = None
        self.labels = self.processor.get_labels()

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollator(self.tokenizer, max_length=self.data_args.max_seq_length, pad_to_multiple_of=8 if pad_to_multiple_of_8 else None, pad_to_max_length=self.data_args.pad_to_max_length)

    def get_examples(self, set_type):
        def read_pseudo(input_file):
            """Reads a json list file."""
            with open(input_file, "r", encoding="utf-8") as f:
                reader = f.readlines()
                lines = []
                for line in reader:
                    pseudo_data = json.loads(line.strip())
                    if float(pseudo_data["pseudo_proba"]) >= self.pseudo_threshold:
                        lines.append(pseudo_data)
                return lines

        examples = list()
        if set_type == "train":
            if self.is_pseudo:
                train_lines = self._read_json2(os.path.join(self.data_dir, "train.json"))
                dev_pseudo_num, test_pseudo_num = 0, 0
                if os.path.exists(os.path.join(self.data_dir, "dev_pseudo.json")):
                    dev_pseudo_lines = read_pseudo(os.path.join(self.data_dir, "dev_pseudo.json"))
                    train_lines.extend(dev_pseudo_lines)
                    dev_pseudo_num += len(dev_pseudo_lines)
                if os.path.exists(os.path.join(self.data_dir, "test_pseudo.json")):
                    test_pseudo_lines = read_pseudo(os.path.join(self.data_dir, "test_pseudo.json"))
                    train_lines.extend(test_pseudo_lines)
                    test_pseudo_num += len(test_pseudo_lines)
                # print("train_lines=", train_lines)
                examples = self._create_examples(train_lines, "train")
                print("add pseudo dev num={}".format(str(dev_pseudo_num)))
                print("add pseudo test num={}".format(str(test_pseudo_num)))
                print("add pseudo all num={}".format(str(dev_pseudo_num + test_pseudo_num)))
                self.train_examples = examples
            else:
                examples = self._create_examples(self._read_json2(self.train_file), "train")
                # examples = examples[:self.data_args.max_train_samples]
                self.train_examples = examples
        elif set_type == "dev":
            examples = self._create_examples(self._read_json2(self.dev_file), "dev")
            examples = examples[:self.data_args.max_eval_samples]
            self.dev_examples = examples
        elif set_type == "test":
            examples = self._create_examples(self._read_json2(self.test_file), "test")
            examples = examples[:self.data_args.max_predict_samples]
            self.test_examples = examples
        return examples

    def _create_examples(self, lines, set_type):
        examples = self.processor.create_examples(lines, set_type)
        return examples

    def build_preprocess_function(self):
        # Tokenize the texts
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length

        def func(examples):
            # Tokenize
            # print("examples["text_b"]=", examples["text_b"])
            if examples["text_b"][0] == None:
                text_pair = None
            else:
                text_pair = examples["text_b"]
            tokenized_examples = tokenizer(
                examples["text_a"],
                text_pair=text_pair,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length" if self.data_args.pad_to_max_length else False,
                return_offsets_mapping=True
            )
            # 确定label
            return tokenized_examples

        return func

    def get_predict_result(self, logits, examples, stage="dev"):
        # logits: [test_data_num, label_num]
        predictions = dict() # 获取概率最大的作为预测结果
        topk_result = dict() # 根据概率取TopK个
        pseudo_data = list() # 根据预测的概率生成伪标签数据
        preds = logits
        if self.output_modes == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.output_modes == "regression":
            preds = np.squeeze(preds)

        for pred, example, logit in zip(preds, examples, logits):
            id_ = example["guid"]
            id_ = int(id_.split("-")[1])
            predictions[id_] = pred # 保存预测结果索引labelid
            # 获取TopK结果
            # {"prob": prob, "answer": answer}
            # print("logit=", logit)
            proba = softmax(logit) # 转换为概率
            # print("proba=", proba)
            # print("========")
            indices = np.argsort(-proba)# 获得降序排列后的索引
            out = list()
            for index in indices[:20]: # 依次取出相应的logit
                prob = proba[index].tolist()
                index = index.tolist()
                out.append({"prob": prob, "answer": index})
            topk_result[id_] = out

            pseudo_proba = proba[pred]
            # pseudo_predicts[id_] = {"label": pred, "pseudo_proba": pseudo_proba}

            # 顺便保存一下pseudo data
            # if pseudo_proba >= 0.99:
            pseudo_data.append({
                "guid": str(id_),
                "text_a": example["text_a"],
                "text_b": example["text_b"],
                "label": str(pred),
                "pseudo_proba": str(pseudo_proba)
            })

        # 保存标签结果
        with open(os.path.join(self.data_dir, "{}_pseudo.json".format(stage)), "w") as writer:
            for i, pred in enumerate(pseudo_data):
                json_d = pred
                writer.write(json.dumps(json_d, ensure_ascii=False) + "\n")

        # if self.data_name == "csl": # acc: 0.9993
        #     if stage == "dev":
        #         abst2guid = dict()
        #         for example in examples:
        #             id_ = example["guid"]
        #             id_ = int(id_.split("-")[1])
        #             abst = example["text_b"]
        #             if abst not in abst2guid.keys():
        #                 abst2guid[abst] = list()
        #             abst2guid[abst].append(id_)
        #         for abst, guids in abst2guid.items():
        #             num = int(len(guids) / 2)
        #             for ei, guid in enumerate(guids):
        #                 label = 0 if ei < num else 1
        #                 predictions[guid] = label
        #         # print(predictions)

        return predictions, topk_result

    def compute_metrics(self, eval_predictions):
        examples = self.raw_datasets["validation"]
        labels = examples["label"]

        golden, dataname_map, dataname_type = {}, defaultdict(list), {}
        predictions, _ = self.get_predict_result(eval_predictions[0], examples, stage="dev") # {"xx": "xxx", ...}
        for example in examples:
            data_type = self.output_modes
            data_name = self.data_name
            if data_name not in dataname_type:
                dataname_type[data_name] = data_type
            id_ = example["guid"]
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
            if len(gold) != len(pred) or len(gold) < 20:
                # print(dataname, dataname_type[dataname], round(acc, 4), len(gold), len(pred), data_ids)
                print("len(gold)=", len(gold))
                print("len(pred)=", len(pred))
            all_metrics["eval_macro_f1"] += f1
            all_metrics["eval_micro_f1"] += f1 * len(data_ids)
            all_metrics["eval_num"] += len(data_ids)
            all_metrics["eval_acc"] += acc
            all_metrics[dataname] = round(f1, 4)
        all_metrics["eval_macro_f1"] = round(all_metrics["eval_macro_f1"] / len(dataname_map), 4)
        all_metrics["eval_micro_f1"] = round(all_metrics["eval_micro_f1"] / all_metrics["eval_num"], 4)
        all_metrics["eval_macro_acc"] = round(all_metrics["eval_acc"] / len(dataname_map), 4)

        return all_metrics

    def save_result(self, logits, label_ids):
        examples = self.raw_datasets["test"]
        predicts, topk_predictions = self.get_predict_result(logits, examples, stage="test")
        clue_processor = clue_processors[self.data_name]()
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

        # outfile = os.path.join(self.training_args.output_dir, "answer.json")
        # with open(outfile, "w", encoding="utf8") as f:
        # #     json.dump(predicts, f, ensure_ascii=False, indent=2)
        #     for res in answer:
        #         f.write("{}\n".format(str(res)))

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



class CSLEFLProcessor(CLUEProcessor):

    def get_predict_result(self, logits, examples, stage="dev"):
        # logits: [test_data_num, label_num]
        predictions = dict() # 获取概率最大的作为预测结果
        topk_result = dict() # 根据概率取TopK个
        preds = logits
        if self.output_modes == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.output_modes == "regression":
            preds = np.squeeze(preds)

        if stage == "train":
            for pred, example, logit in zip(preds, examples, logits):
                id_ = example["guid"] # 验证集和训练集一样，每个abst文本与一个关键词组成example，所以这里的guid并不是原始数据集的编号
                id_ = int(id_.split("-")[1])
                predictions[id_] = pred # 保存预测结果索引labelid
                # 获取TopK结果
                # {"prob": prob, "answer": answer}
                # print("logit=", logit)
                proba = softmax(logit) # 转换为概率
                # print("proba=", proba)
                # print("========")
                indices = np.argsort(-proba)# 获得降序排列后的索引
                out = list()
                for index in indices[:20]: # 依次取出相应的logit
                    prob = proba[index].tolist()
                    index = index.tolist()
                    out.append({"prob": prob, "answer": index})
                topk_result[id_] = out
        else:
            # pred_keywords = dict() # 每个example对应的关键词列表
            pred_probas = dict() # 每个examp对应所有预测结果的概率
            pred_keys = dict()
            for pred, example, logit in zip(preds, examples, logits):
                id_ = example["guid"] # 测试集的guid与原始数据集对应
                id_ = int(id_.split("-")[1])
                text_a = example["text_a"]
                # 模板为"【{}】可以作为文章的关键词吗？文章：【{}】"，获取关键词，则只需要前几个token即可
                keyword = text_a.split("】")[0][1:]
                proba = softmax(logit)  # 转换为概率
                if id_ not in pred_probas.keys():
                    # pred_keywords[id_] = list()
                    pred_probas[id_] = list()
                    pred_keys[id_] = list()
                pred_probas[id_].append(proba[1])
                pred_keys[id_].append(keyword)
                # if id_ not in predictions.keys():
                #     # 如果首次预测编号为id的example，暂时先设置最终预测结果为1，等待后续如果出现关键词预测为0的进行处理
                #     predictions[id_] = 1
                # if pred == 0:
                #     # 一旦出现了预测为0，即当前编号id_的example出现了预测为不正确的关键词了，直接将当前的预测结果置为"0"
                #     # 如果当前example的所有关键词都被预测为1，则一直不会执行该部分，所以最终预测结果一直保持为1
                #     predictions[id_] = 0
            # 还需要获得每个example的预测概率，直接把涉及到的所有关键词的预测概率进行平均即可
            for id_, probas in pred_probas.items():
                key_length_num = 0
                keys = pred_keys[id_]
                for key in keys:
                    if len(key) == 2:
                        key_length_num += 1
                print("id={}, probas={}".format(id_, " ".join([str(prob) for prob in probas])))
                prob = np.min(np.array(probas), -1).tolist() # 寻找短板预测结果
                # 注意，因为概率都是根据每个关键词得到的，因此如果预测的结果为1，均值一定大于等于0.5，但是均值大于0.5不一定
                # 代表当前example预测为1，例如，可能当前example的5个关键词有1个预测概率为0.4，而其余的为0.9，均值大于0.5，但是
                # 存在了一个错误的关键词，导致当前预测结果为0。因此我们选择所有关键词中预测概率最小的作为每个example的预测得分
                # 因此，预测概率最小的如果大于等于0.5，则说明当前example的预测为1，否则为0，下面assert可以check一下
                # assert predictions[id_] == 1 and prob >= 0.5
                # assert predictions[id_] == 0 and prob < 0.5
                predictions[id_] = 1 if prob >= 0.5 else 0
                # if key_length_num >= 2:
                #     predictions[id_] = 0
                if prob >= 0.5:
                    topk_result[id_] = [
                        {"prob": prob, "answer": 1},
                        {"prob": 1 - prob, "answer": 0}
                    ]
                else:
                    topk_result[id_] = [
                        {"prob": 1 - prob, "answer": 0},
                        {"prob": prob, "answer": 1}
                    ]
        return predictions, topk_result





# 为Tnews数据集定制的EFL方案
# 原始Tnews为15类分类，直接让模型预测15个类。EFL版本则转换为15个二分类
class TnewsEFLProcessor(CLSProcessor):
    def __init__(self, data_args, training_args, model_args, tokenizer=None, post_tokenizer=False, keep_raw_data=True):
        super().__init__(data_args, training_args, model_args, tokenizer, post_tokenizer=post_tokenizer, keep_raw_data=keep_raw_data)
        param = {p.split("=")[0]: p.split("=")[1] for p in (data_args.user_defined).split(" ")}
        assert "data_name" in param, "You must add one defined param 'data_name=xxx' in the user_defined parameter."
        self.data_name = param["data_name"]
        # assert self.data_name in clue_processors.keys(), "Unknown task name {}".format(self.data_name)
        self.processor = clue_processors["tnews_efl"]()
        self.output_modes = clue_output_modes[self.data_name]
        self.label_desc = {
            "100": "故事", "101": "文学", "102": "娱乐", "103": "体育", "104": "财金", "106": "家居",
            "107": "汽车", "108": "教育", "109": "科技", "110": "军事", "112": "旅游", "113": "国际",
            "114": "股票", "115": "农业", "116": "游戏"
        }
        self.train_file = os.path.join(data_args.data_dir, "train.json")
        self.dev_file = os.path.join(data_args.data_dir, "dev.json")
        self.test_file = os.path.join(data_args.data_dir, "test.json")
        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.sentence1_key = None
        self.labels = self.processor.get_labels()

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollator(self.tokenizer, pad_to_multiple_of=8 if pad_to_multiple_of_8 else None, pad_to_max_length=self.data_args.pad_to_max_length)

    def get_examples(self, set_type):
        examples = list()
        if set_type == "train":
            examples = self._create_examples(self._read_json2(self.train_file), "train", label_desc=self.label_desc)
            examples = examples[:self.data_args.max_train_samples]
            self.train_examples = examples
        elif set_type == "dev":
            examples = self._create_examples(self._read_json2(self.dev_file), "dev", label_desc=self.label_desc)
            examples = examples[:self.data_args.max_eval_samples]
            self.dev_examples = examples
        elif set_type == "test":
            examples = self._create_examples(self._read_json2(self.test_file), "test", label_desc=self.label_desc)
            examples = examples[:self.data_args.max_predict_samples]
            self.test_examples = examples
        return examples

    def _create_examples(self, lines, set_type, label_desc):
        examples = self.processor.create_examples(lines, set_type, label_desc=label_desc)
        return examples

    def set_config(self, config):
        config.ent_type_size = 1
        config.inner_dim = 64
        config.RoPE = True

    def build_preprocess_function(self):
        # Tokenize the texts
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length

        def func(examples):
            # Tokenize
            # print("examples["text_b"]=", examples["text_b"])
            if examples["text_b"][0] == None:
                text_pair = None
            else:
                text_pair = examples["text_b"]
            tokenized_examples = tokenizer(
                examples["text_a"],
                text_pair=text_pair,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length" if self.data_args.pad_to_max_length else False,
                return_offsets_mapping=True
            )
            # 确定label
            return tokenized_examples

        return func

    def get_predict_result(self, logits, examples):
        # logits: [test_data_num, label_num]
        predictions = dict() # 获取概率最大的作为预测结果
        topk_result = dict() # 根据概率取TopK个
        preds = logits
        if self.output_modes == "classification":
            preds = np.argmax(preds, axis=1)
        elif self.output_modes == "regression":
            preds = np.squeeze(preds)

        for pred, example, logit in zip(preds, examples, logits):
            id_ = example["guid"]
            id_ = int(id_.split("-")[1])
            predictions[id_] = pred # 保存预测结果
            # 获取TopK结果
            # {"prob": prob, "answer": answer}
            indices = np.argsort(-logit)# 获得降序排列后的索引
            out = list()
            for index in indices[:20]: # 依次取出相应的logit
                prob = logit[index]
                out.append({"prob": prob, "answer": index})
            topk_result[id_] = out

        return predictions, topk_result

    def compute_metrics(self, eval_predictions):
        examples = self.raw_datasets["validation"]
        labels = examples["label"]


        golden, dataname_map, dataname_type = {}, defaultdict(list), {}
        predictions, _ = self.get_predict_result(eval_predictions[0], examples) # {"xx": "xxx", ...}
        for example in examples:
            data_type = self.output_modes
            data_name = self.data_name
            if data_name not in dataname_type:
                dataname_type[data_name] = data_type
            id_ = example["guid"]
            id_ = int(id_.split("-")[1])
            dataname_map[data_name].append(id_)
            golden[id_] = example["label"]

        all_metrics = {
            "macro_f1": 0.,
            "micro_f1": 0.,
            "eval_num": 0,
            "acc": 0.,
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
            all_metrics["macro_f1"] += f1
            all_metrics["micro_f1"] += f1 * len(data_ids)
            all_metrics["eval_num"] += len(data_ids)
            all_metrics["acc"] += acc
            all_metrics[dataname] = round(acc, 4)
        all_metrics["macro_f1"] = round(all_metrics["macro_f1"] / len(dataname_map), 4)
        all_metrics["micro_f1"] = round(all_metrics["micro_f1"] / all_metrics["eval_num"], 4)
        all_metrics["macro_acc"] = round(all_metrics["acc"] / len(dataname_map), 4)

        return all_metrics

    def save_result(self, logits, label_ids):
        examples = self.raw_datasets["test"]
        predicts, topk_predictions = self.get_predict_result(logits, examples)
        clue_processor = clue_processors[self.data_name]()
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

        # outfile = os.path.join(self.training_args.output_dir, "answer.json")
        # with open(outfile, "w", encoding="utf8") as f:
        # #     json.dump(predicts, f, ensure_ascii=False, indent=2)
        #     for res in answer:
        #         f.write("{}\n".format(str(res)))

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
        with open(topfile, "w", encoding="utf8") as f2:
            json.dump(topk_predictions, f2, ensure_ascii=False, indent=4)
