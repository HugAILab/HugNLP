# -*- coding: utf-8 -*-
# @Time    : 2022/2/21 11:38 上午
# @Author  : JianingWang
# @File    : conll2003
import numpy as np
from datasets import load_dataset, load_metric
from transformers import DataCollatorForTokenClassification, EvalPrediction
from processors.ProcessorBase import DataProcessor


class Conll2003Processor(DataProcessor):

    def __init__(self, data_args, training_args, model_args):
        super().__init__(data_args, training_args, model_args)

    def get_examples(self, set_type):
        raw_datasets = load_dataset(
            'conll2003'
        )
        return raw_datasets

    def get_data_collator(self):
        return DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8 if self.training_args.fp16 else None)

    def get_tokenized_datasets(self):
        return NotImplementedError()

    def compute_metrics(self, p: EvalPrediction):
        metric = load_metric("seqeval")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

