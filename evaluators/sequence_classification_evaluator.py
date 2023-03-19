# -*- coding: utf-8 -*-
# @Time    : 2023/3/5 11:10 下午
# @Author  : JianingWang
# @File    : EvaluatorBase.py

import json
import os.path
import math
import numpy as np
from typing import Dict, Union, Any, Optional, Callable, List, Tuple, Iterator
import datasets
from datasets import Dataset
from config import DataTrainingArguments, TrainingArguments
from hugnlp_trainer import HugTrainer
from processors.ProcessorBase import DataProcessor
from evaluators.EvaluatorBase import NO_GENERATE, DO_GENERATE, Evaluator, ClassificationEvaluator
from metrics.classification_metric import ClassificationMetric
from tools.runner_utils.log_util import logging

logger = logging.getLogger(__name__)


"""
Evaluator for the task of sequence classification with Masked PLMs.
"""
class SequenceClassificationEvaluator(ClassificationEvaluator):

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



"""
Evaluator for the task of sequence classification with Causal PLMs.
"""
class CausalSequenceClassificationEvaluator(Evaluator):

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
        self.paradigm = DO_GENERATE
