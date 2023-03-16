# -*- coding: utf-8 -*-
# @Time    : 2023/3/5 11:10 下午
# @Author  : JianingWang
# @File    : EvaluatorBase.py

import json
import os.path
import math
import numpy as np
from collections import defaultdict
from typing import Dict, Union, Any, Optional, Callable, List, Tuple, Iterator
import datasets
from datasets import Dataset
from config import DataTrainingArguments, TrainingArguments
from hugnlp_trainer import HugTrainer
from processors.ProcessorBase import DataProcessor
from metrics.classification_metric import ClassificationMetric
from tools.runner_utils.log_util import logging

logger = logging.getLogger(__name__)

DO_GENERATE = "do_generate" # mark the task as generate
NO_GENERATE = "no_generate" # mark the task as non-generate


class Evaluator(object):

    def __init__(
        self,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        processor: DataProcessor,
        trainer: Optional[HugTrainer],
        eval_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> None:

        self.data_args = data_args
        self.training_args = training_args
        self.processor = processor
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.model = trainer.model
        self.paradigm = None # You must define this as DO_GENERATE or NO_GENERATE when inherit this class.
        self.metric = None # You must define the metric when inherit this class.

    def default_compute_metrics(self, eval_predictions):
        """
        Design for the default metrics calculation for the current task.
        Note: If the task processor has attribution of 'compute_metrics', this function will not be used.
        """
        pass


    def evaluate(self):
        """
        Design for validation.
        """
        pass


    def test(self):
        """
        Design for testing if the testing examples have labels.
        """
        pass


    def predict(self):
        """
        Design for testing if the testing examples have no labels.
        """
        pass


    def get_topk(self):
        """
        Obtain Top K predictions.
        """
        pass



"""
Evaluator for the task of sequence classification.
"""
class ClassificationEvaluator(Evaluator):

    def __init__(
        self,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        processor: DataProcessor,
        trainer: Optional[HugTrainer],
        eval_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(data_args, training_args, processor, trainer, eval_dataset, test_dataset)
        self.paradigm = NO_GENERATE


    def get_predict_result(self, logits, examples, stage="dev"):
        # logits: [test_data_num, label_num]
        predictions = dict()  # 获取概率最大的作为预测结果
        topk_result = dict()  # 根据概率取TopK个
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

    def default_compute_metrics(self, eval_predictions):
        """
        Design for the default metrics calculation for the current task.
        Note: If the task processor has attribution of 'compute_metrics', this function will not be used.
        """
        examples = self.eval_dataset
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
            metric = ClassificationMetric()
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

    def evaluate(self):

        # If has no compute_metrics for HugTrainer, we can choose default function.
        if not hasattr(self.trainer, "compute_metrics") or self.trainer.compute_metrics is None:


        metrics = self.trainer.evaluate()
        max_eval_samples = self.data_args.max_eval_samples if self.data_args.max_eval_samples is not None else len(self.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))
        if self.data_args.task_type == "mlm":
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

    def predict(self):

        if not self.data_args.keep_predict_labels:
            for l in ["labels", "label"]:
                if l in test_dataset.column_names:
                    test_dataset = test_dataset.remove_columns(l)

        prediction = self.trainer.predict(test_dataset, metric_key_prefix="predict")
        logits = prediction.predictions

        if self.data_args.keep_predict_labels:
            label_ids = prediction.label_ids

        if hasattr(self.processor, "save_result"):
            if self.trainer.is_world_process_zero():
                if not self.data_args.keep_predict_labels:
                    self.processor.save_result(logits)
                else:
                    self.processor.save_result(logits, label_ids)
        else:
            predictions = np.argmax(logits, axis=1)
            output_predict_file = os.path.join(self.training_args.output_dir, f"predict_results.txt")
            if self.trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {self.data_args.task_name} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = self.processor.labels[item]
                        writer.write(f"{index}\t{item}\n")
