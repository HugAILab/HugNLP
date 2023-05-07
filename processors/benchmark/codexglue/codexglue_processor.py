'''
# -*- coding: utf-8 -*-
Author: nchen909 NuoChen
Date: 2023-05-06 16:16:16
FilePath: /HugNLP/processors/benchmark/codexglue/codexglue_processor.py
'''
"""Dataset utils for different data settings for GLUE."""

import os
import copy
import logging
import torch
import numpy as np
import time
from filelock import FileLock
import json
import itertools
import random
import transformers
from processors.benchmark.codexglue.utils import DataProcessor, InputExample, InputFeatures, DefectExample, CloneExample
from transformers.data.processors.glue import *
from transformers.data.metrics import acc_and_f1
import dataclasses
from dataclasses import dataclass, asdict
from typing import List, Optional, Union
# from sentence_transformers import SentenceTransformer, util
from copy import deepcopy
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class BigCloneBenchProcessor(DataProcessor):
    """Processor for the BigCloneBench data set (CodeXGLUE version)."""
    def __init__(self):
        pass

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["id"].numpy(),
            tensor_dict["func1"].numpy().decode("utf-8"),
            tensor_dict["func2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=text_b,
                             label=label))
        return examples

class DevignProcessor(DataProcessor):
    """Processor for the Devign data set (CodeXGLUE version)."""
    def __init__(self):
        pass

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["id"].numpy(),
            tensor_dict["func1"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[text_index]
            label = line[1]
            examples.append(
                InputExample(guid=guid,
                             text_a=text_a,
                             text_b=None,
                             label=label))
        return examples



class TextClassificationProcessor(DataProcessor):
    """Data processor for text classification datasets (mr, sst-5, subj, trec, cr, mpqa)."""
    def __init__(self, task_name):
        self.task_name = task_name

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_csv(os.path.join(data_dir, "train.csv"),
                        header=None).values.tolist(), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_csv(os.path.join(data_dir, "dev.csv"),
                        header=None).values.tolist(), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            pd.read_csv(os.path.join(data_dir, "test.csv"),
                        header=None).values.tolist(), "test")

    def get_labels(self):
        """See base class."""
        if self.task_name == "mr":
            return list(range(2))
        elif self.task_name == "sst-5":
            return list(range(5))
        elif self.task_name == "subj":
            return list(range(2))
        elif self.task_name == "trec":
            return list(range(6))
        elif self.task_name == "cr":
            return list(range(2))
        elif self.task_name == "mpqa":
            return list(range(2))
        else:
            raise Exception("task_name not supported.")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if self.task_name == "ag_news":
                examples.append(
                    InputExample(guid=guid,
                                 text_a=line[1] + ". " + line[2],
                                 short_text=line[1] + ".",
                                 label=line[0]))
            elif self.task_name == "yelp_review_full":
                examples.append(
                    InputExample(guid=guid,
                                 text_a=line[1],
                                 short_text=line[1],
                                 label=line[0]))
            elif self.task_name == "yahoo_answers":
                text = line[1]
                if not pd.isna(line[2]):
                    text += " " + line[2]
                if not pd.isna(line[3]):
                    text += " " + line[3]
                examples.append(
                    InputExample(guid=guid,
                                 text_a=text,
                                 short_text=line[1],
                                 label=line[0]))
            elif self.task_name in [
                    "mr", "sst-5", "subj", "trec", "cr", "mpqa"
            ]:
                examples.append(
                    InputExample(guid=guid, text_a=line[1], label=line[0]))
            else:
                raise Exception("Task_name not supported.")

        return examples


def text_classification_metrics(task_name, preds, labels):
    return {"acc": (preds == labels).mean()}


# Add your task to the following mappings

# task_to_test_key = {
#     "cola": "matt",
#     "mnli": "accuracy",
#     "mrpc": "f1",
#     "qnli": "accuracy",
#     "qqp": "f1",
#     "rte": "accuracy",
#     "sst2": "accuracy",
#     "stsb": "accuracy",
#     "wnli": "accuracy",
# }

task_to_keys = {
    "bcb": ("func1", "func2"),
    "devign": ("func1", None),
}

codexglue_processors = {
    "bcb": BigCloneBenchProcessor,
    "devign": DevignProcessor,
}

num_labels_mapping = {
    "bcb": 2,
    "devign": 2,
}

output_modes_mapping = {
    "bcb": "classification",
    "devign": "classification",
}

# Return a function that takes (task_name, preds, labels) as inputs
compute_metrics_mapping = {
    "bcb": acc_and_f1,
    "devign": acc_and_f1,
}
