# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 8:09 p.m.
# @Author  : JianingWang
# @File    : reinforcement_learning_evaluator.py

import json
import os.path
import math
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Union, Any, Optional, Callable, List, Tuple, Iterator
import datasets
from datasets import Dataset
from config import DataTrainingArguments, TrainingArguments, ModelArguments
from hugnlp_trainer import HugTrainer
from processors.ProcessorBase import DataProcessor
from evaluators.EvaluatorBase import NO_GENERATE, DO_GENERATE, Evaluator, ClassificationEvaluator, GenerationEvaluator
from metrics.classification_metric import ClassificationMetric
from tools.runner_utils.log_util import logging
from tools.computations.softmax import softmax
from tools.model_utils.calibrate import CausalCLSCalibrator

logger = logging.getLogger(__name__)


"""
Evaluator for pair-wise reward model
"""
class PairwiseRewardEvaluator(ClassificationEvaluator):

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


    def evaluate(self, test_dataset=None):

        """
        Each example has following two sequence:
        - chosen: the better response
        - rejected: the worse response
        We need the model assign high reward for chosen than rejected sequence.
        Thus, we calculate the accuracy that the reward value of chosen sequence derived from the reward model higher than the rejected sequence.
        """
        eval_dataset = self.eval_dataset if test_dataset is not None else test_dataset
        all_chosen_values, all_rejected_values = list(), list()
        for ei, data in enumerate(tqdm(eval_dataset)):
            # chosen_input_ids, chosen_attention_mask = data["chosen_sequence"], data["chosen_attention_mask"]
            # rejected_input_ids, rejected_attention_mask = data["rejected_sequence"], data["rejected_attention_mask"]
            chosen_output = self.model(**data)
            chosen_values, rejected_values = chosen_output["chosen_values"], chosen_output["rejected_values"]
            all_chosen_values.extend(chosen_values.detach().cpu().numpy().tolist())
            all_rejected_values.extend(rejected_values.detach().cpu().numpy().tolist())

        metrics = dict()
        acc = 0.
        for chosen_value, rejected_value in zip(all_chosen_values, all_rejected_values):
            if chosen_value >= rejected_value:
                acc += 1
        metrics["acc"] = round(acc / len(all_chosen_values), 4)
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)


    def predict(self):

        self.evaluate(test_dataset=self.test_dataset)

    def get_best_and_topk(self, logits, examples, topk=10, stage="dev"):
        pass
