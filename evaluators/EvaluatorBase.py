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
import torch
import datasets
from datasets import Dataset
from config import DataTrainingArguments, TrainingArguments, ModelArguments
from hugnlp_trainer import HugTrainer
from processors.ProcessorBase import DataProcessor
from metrics.classification_metric import ClassificationMetric
from tools.computations.softmax import softmax
from tools.runner_utils.log_util import logging
from tools.model_utils.gpt_response import GPTResponse

logger = logging.getLogger(__name__)

DO_GENERATE = "do_generate" # mark the task as generate
NO_GENERATE = "no_generate" # mark the task as non-generate


class Evaluator(object):

    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        processor: DataProcessor,
        trainer: Optional[HugTrainer],
        eval_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> None:

        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.processor = processor
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.model = trainer.model
        self.paradigm = None # You must define this as DO_GENERATE or NO_GENERATE when inherit this class.
        self.metric = None # You must define the metric when inherit this class.

    def reset_trainer(self, trainer: HugTrainer):
        logger.info("Reset trainer state in evaluation.")
        self.trainer = trainer

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


    def get_best_and_topk(self):
        """
        Obtain the best results and Top K predictions.
        """
        pass



"""
Evaluator for classification-style tasks.
"""
class ClassificationEvaluator(Evaluator):

    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        processor: DataProcessor,
        trainer: Optional[HugTrainer],
        eval_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(model_args, data_args, training_args, processor, trainer, eval_dataset, test_dataset)
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


    def evaluate(self):

        # If has no compute_metrics for HugTrainer, we can choose default function.
        if not hasattr(self.trainer, "compute_metrics") or self.trainer.compute_metrics is None:
            self.trainer.compute_metrics = self.default_compute_metrics

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
            label_list = self.processor.labels
            id2label = {i: label for i, label in enumerate(label_list)}

            # submit
            answer = list()
            for k, v in predicts.items():
                if v not in id2label.keys():
                    res = ""
                    print("unknown")
                else:
                    res = id2label[v]
                answer.append({"id": k, "label": res})

            output_submit_file = os.path.join(self.training_args.output_dir, "answer.json")
            # Save the label results
            with open(output_submit_file, "w") as writer:
                for i, pred in enumerate(answer):
                    json_d = {}
                    json_d["id"] = i
                    json_d["label"] = pred["label"]
                    writer.write(json.dumps(json_d) + "\n")

            # Save Top K results
            topfile = os.path.join(self.training_args.output_dir, "topk_predict.json")
            with open(topfile, "w", encoding="utf-8") as f2:
                json.dump(topk_predictions, f2, ensure_ascii=False, indent=4)


    def get_best_and_topk(self, logits, examples, topk=10, stage="dev"):
        """
        Obtain the best results and Top K predictions.
        """
        if type(logits) == tuple:
            logits = logits[0]
        # logits: [test_data_num, label_num]
        predictions = dict() # Obtain the best predictions
        topk_result = dict() # Obtain the Top K predictions

        preds = logits
        preds = np.argmax(preds, axis=1)

        for pred, example, logit in zip(preds, examples, logits):
            id_ = example["idx"]
            id_ = int(id_.split("-")[1])
            predictions[id_] = pred
            proba = softmax(logit) # Transform as probabilities.
            indices = np.argsort(-proba)
            out = list()
            for index in indices[:topk]:
                prob = proba[index].tolist()
                index = index.tolist()
                out.append({"prob": prob, "answer": index})
            topk_result[id_] = out

        return predictions, topk_result


"""
Evaluator for generation-style tasks.
"""
class GenerationEvaluator(Evaluator):

    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        processor: DataProcessor,
        trainer: Optional[HugTrainer],
        eval_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> None:
        super().__init__(model_args, data_args, training_args, processor, trainer, eval_dataset, test_dataset)
        self.paradigm = NO_GENERATE
        self.generation_model_response = None
        if self.model_args.model_type in ["gpt2", "gpt3"]:
            self.generation_model_response = GPTResponse(
                model_type=self.model_args.model_type,
                data_path=self.data_args.data_dir
            )


    def default_compute_metrics(self, eval_predictions):
        pass
        # return all_metrics

    def generate(self, eval_dataset, num_log_probs=1, echo=False):
        # obtain generative answer from causal PLM.
        assert hasattr(self.processor, "l"), "You must define l ('max length of generated tokens') for generative-style tasks"
        l = self.processor.l

        all_raw_answers = []

        return all_raw_answers





    def evaluate(self):
        pass


    def predict(self):

        pass


    def get_best_and_topk(self, logits, examples, topk=10, stage="dev"):
        """
        Obtain the best results and Top K predictions.
        """


        pass
