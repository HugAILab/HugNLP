# -*- coding: utf-8 -*-
# @Time    : 2023/3/5 11:10 下午
# @Author  : JianingWang
# @File    : EvaluatorBase.py

import json
import os.path
import math
import torch
import numpy as np
from typing import Dict, Union, Any, Optional, Callable, List, Tuple, Iterator
import datasets
from datasets import Dataset
from config import DataTrainingArguments, TrainingArguments, ModelArguments
from hugnlp_trainer import HugTrainer
from processors.ProcessorBase import DataProcessor
from evaluators.EvaluatorBase import NO_GENERATE, DO_GENERATE, Evaluator, ClassificationEvaluator
from metrics.classification_metric import ClassificationMetric
from tools.runner_utils.log_util import logging

logger = logging.getLogger(__name__)


"""
Evaluator for the task of span extraction with Masked PLMs.
"""
class SpanExtractionEvaluator(ClassificationEvaluator):

    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        processor: DataProcessor,
        model: torch.nn.Module,
        trainer: Optional[HugTrainer] = None,
        eval_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(model_args, data_args, training_args, processor, model, trainer, eval_dataset, test_dataset)
        self.paradigm = NO_GENERATE


    def default_compute_metrics(self, eval_predictions):
        """
        Design for the default metrics calculation for the current task.
        Note:
        - If the task processor has attribution of 'compute_metrics', this function will not be used.
        - If this pre-built function can match your demand, you can omit the definition of 'compute_metrics' in your processor.
        """
        examples = self.eval_dataset
        labels = examples["label"]

        golden = {}
        # predictions:  {"xx": "xxx", ...}
        predictions, _ = self.get_best_and_topk(eval_predictions[0], examples, stage="dev")
        for example in examples:
            try:
                idx = int(example["idx"])
            except:
                idx = int(example["idx"].split("-")[1]) # e.g.,  "dev-12" -> "12"

            golden[idx] = example["label"]

        all_metrics = {
            "eval_macro_f1": 0.,
            "eval_acc": 0.,
        }

        metric = ClassificationMetric()
        gold = {k: v for k, v in golden.items()}
        pred = {k: v for k, v in predictions.items()}
        score = metric.calc_metric(golden=gold, predictions=pred)
        acc, f1 = score["acc"], score["f1"]
        all_metrics["eval_macro_f1"] += f1
        all_metrics["eval_acc"] += acc
        return all_metrics


    def predict(self):

        assert self.paradigm == NO_GENERATE, "classification only support no-generate model."
        if not self.data_args.keep_predict_labels:
            for l in ["labels", "label"]:
                if l in self.test_dataset.column_names:
                    self.test_dataset = self.test_dataset.remove_columns(l)

        prediction = self.trainer.predict(self.test_dataset, metric_key_prefix="predict")
        logits = prediction.predictions

        if self.data_args.keep_predict_labels:
            label_ids = prediction.label_ids

        # If you have defined save_result function in the processor.
        if hasattr(self.processor, "save_result"):
            assert self.paradigm == NO_GENERATE, "default processor only support no-generate model."
            if self.trainer.is_world_process_zero():
                if not self.data_args.keep_predict_labels:
                    self.processor.save_result(logits)
                else:
                    self.processor.save_result(logits, label_ids)
        else:
            # If you not define the save_result function.
            examples = self.test_dataset
            predicts, topk_predictions = self.get_best_and_topk(logits, examples, stage="test")
            outfile = os.path.join(self.training_args.output_dir, "answer.json")

            with open(outfile, "w", encoding="utf8") as f:
                json.dump(predicts, f, ensure_ascii=False, indent=2)

            topk_file = os.path.join(self.training_args.output_dir, "topk_predict.json")
            with open(topk_file, "w", encoding="utf8") as f2:
                json.dump(topk_predictions, f2, ensure_ascii=False, indent=2)



    def get_best_and_topk(self, logits, examples, topk=10, stage="dev"):

        def fush_multi_answer(self, has_answer, new_answer):
            # has {"ans": {"prob": float(prob[index_ids[ei]]), "pos": (s, e)}, ...}
            # new {"ans": {"prob": float(prob[index_ids[ei]]), "pos": (s, e)}, ...}
            for ans, value in new_answer.items():
                if ans not in has_answer.keys():
                    has_answer[ans] = value
                else:
                    has_answer[ans]["prob"] += value["prob"]
                    has_answer[ans]["pos"].extend(value["pos"])
            return has_answer

        probs, indices = logits
        probs = probs.squeeze(1)  # topk probability [n, m]
        indices = indices.squeeze(1)  # topk index [n, m]
        predictions = {}
        topk_predictions = {}
        for prob, index, example in zip(probs, indices, examples):
            id_ = example["idx"]
            index_ids = torch.Tensor([i for i in range(len(index))]).long()
            best_start_end = np.unravel_index(index[0], (self.data_args.max_seq_length, self.data_args.max_seq_length))
            s = example["offset_mapping"][best_start_end[0]][0]
            e = example["offset_mapping"][best_start_end[1]][1]
            answer = example["content"][s: e]
            predictions[id_] = answer

            topk_answer_dict = dict()
            topk_index = index[prob > 0.0]
            index_ids = index_ids[prob > 0.0]
            for ei, index in enumerate(topk_index):
                if ei > topk:
                    break
                start_end = np.unravel_index(index, (self.data_args.max_seq_length, self.data_args.max_seq_length))
                s = example["offset_mapping"][start_end[0]][0]
                e = example["offset_mapping"][start_end[1]][1]
                ans = example["content"][s: e]
                topk_answer_dict[ans] = {"prob": float(prob[index_ids[ei]]), "pos": [(s, e)]}

            predictions[id_] = answer
            if id_ not in topk_predictions.keys():
                topk_predictions[id_] = topk_answer_dict
            else:
                topk_predictions[id_] = fush_multi_answer(topk_predictions[id_], topk_answer_dict)

        for id_, values in topk_predictions.items():
            answer_list = list()
            for ans, value in values.items():
                answer_list.append({"answer": ans, "prob": value["prob"], "pos": value["pos"]})
            topk_predictions[id_] = answer_list

        return predictions, topk_predictions
