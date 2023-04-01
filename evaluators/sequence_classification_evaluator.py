# -*- coding: utf-8 -*-
# @Time    : 2023/3/5 11:10 下午
# @Author  : JianingWang
# @File    : EvaluatorBase.py

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
Evaluator for the task of sequence classification with Masked PLMs.
"""
class SequenceClassificationEvaluator(ClassificationEvaluator):

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



"""
Evaluator for the task of sequence classification with Causal PLMs.
"""
class CausalSequenceClassificationEvaluator(GenerationEvaluator):

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
        self.paradigm = DO_GENERATE
        assert hasattr(self.processor, "label_words_mapping"), "If you choose causal PLM to generate label for classification, you must define 'label_words_mapping'"
        self.label_words_mapping = self.processor.label_words_mapping
        self.label2id = self.processor.label2id
        self.calibrator = CausalCLSCalibrator(self.model, self.processor.tokenizer)


    def default_compute_metrics(self, predictions, examples):

        labels = examples["label"]

        golden = {}
        # predictions:  {"xx": "xxx", ...}
        predictions, _ = self.get_best_and_topk(predictions, examples, stage="dev")
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

        # generate answer for validation set
        all_raw_answers = self.generate(
            eval_dataset=self.eval_dataset,
            num_log_probs=100,
            echo=False, # echo is False means do not directly obtain label probability.
        )

        # obtain the logits of the generated answer for each label.
        all_label_probs = self.obtain_label_logits(all_raw_answers, self.eval_dataset)

        print("all_label_probs=", all_label_probs)

        # calibrate the prediction
        if self.processor.use_calibrate:
            logger.info("Calibrating ...")
            content_free_examples = self.processor.get_content_free_examples()
            all_label_probs = self.calibrator.calibrate(
                all_label_probs=all_label_probs,
                content_free_examples=content_free_examples,
                label2id=self.processor.label2id,
            )
        # print("all_calibrate_label_probs=", all_label_probs)

        metrics = self.default_compute_metrics(all_label_probs, self.eval_dataset)
        # print("dev metrics=", metrics)

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)


    def predict(self):

        # generate answer for test dataset
        all_raw_answers = self.generate(
            eval_dataset=self.test_dataset,
            num_log_probs=100,
            echo=False
        )

        # obtain the logits of the generated answer for each label.
        logits = self.obtain_label_logits(all_raw_answers, self.test_dataset)
        # calibrate the prediction
        if self.processor.use_calibrate:
            logger.info("Calibrating ...")
            content_free_examples = self.processor.get_content_free_examples()
            logits = self.calibrator.calibrate(
                all_label_probs=logits,
                content_free_examples=content_free_examples,
                label2id=self.processor.label2id,
            )
        if "label" in self.test_dataset.features and not self.data_args.keep_predict_labels:

            metrics = self.default_compute_metrics(logits, self.test_dataset)
            # print("test metrics=", metrics)

            self.trainer.log_metrics("test", metrics)
            self.trainer.save_metrics("test", metrics)

        # If you have defined save_result function in the processor.
        if hasattr(self.processor, "save_result"):
            assert self.paradigm == NO_GENERATE, "default processor only support no-generate model."
            if self.trainer.is_world_process_zero():
                if "label" not in self.test_dataset.features:
                    self.processor.save_result(logits)
                else:
                    self.processor.save_result(logits, self.test_dataset["label"])
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


    def generate(self, eval_dataset, num_log_probs=100, echo=False):
        # obtain generative answer from causal PLM.
        assert hasattr(self.processor, "l"), "You must define l ('max length of generated tokens') for generative-style tasks"
        l = self.processor.l
        all_raw_answers = []
        for data in tqdm(eval_dataset):
            total_sequences = self.model.generate(
                input_ids=torch.Tensor([data['input_ids']]).long().to(self.model.device),
                attention_mask=torch.Tensor([data['attention_mask']]).long().to(self.model.device),
                max_length=l + len(data['input_ids']),
                do_sample=False, # If for cls, 'do_sample' must set False
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            # print("total_sequences=", self.processor.tokenizer.convert_ids_to_tokens(total_sequences[0][len(data['input_ids']):])) # e.g. ['Ġnegative']
            if num_log_probs is not None:
                # we are left padding, so we need to adjust the position IDs
                attention_mask = (total_sequences != 50256).float()
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                # get the logits for the context and the next l tokens
                logits = self.model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu()
                response_res = self.generation_model_response.call_for_gpt2_response(self.processor.tokenizer, logits, total_sequences, l, num_log_probs, echo)

                for answer_id, answer in enumerate(response_res['choices']):
                    all_raw_answers.append(answer)

        return all_raw_answers


    def obtain_label_logits(self, answers, examples):
        """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
        """
        answers:
        [{
            'text': ' I', # the top 1 generated token
            'logprobs': {'top_logprobs': [{' I': -3.4267239570617676, '\n': -3.5073862075805664, ...], # Top k tokens at this position
            'token_logprobs': [-3.4267239570617676], # the top 1 generated token score
            'tokens': [' I']} # the generated token list
        }, ...]
        """
        assert len(answers) == len(examples)

        # Fill in the labels that is in the top k prob
        all_label_probs = []
        all_missing_positions = []
        for i, ans in enumerate(answers):
            top_logprobs = ans['logprobs']['top_logprobs'][0]  # [0] since we only ask for complete one more token, dict
            # print("top_logprobs=", top_logprobs)
            # top_logprobs = {' I': -3.4267239570617676, '\n': -3.5073862075805664, ...}
            label_probs = [0.] * len(self.label_words_mapping.keys())
            for j, label_list in self.label_words_mapping.items():
                j = self.label2id[j] # j is the original label name, it must be converted to label id
                all_found = True
                for label in label_list:  # each possible label correspond to the same class
                    label = " " + label  # notice prompt does not have space after 'A:'
                    if label in top_logprobs:
                        label_probs[j] += np.exp(top_logprobs[label])
                    else:
                        all_found = False
                if not all_found:
                    position = (i, j) # (which test example, which label)
                    all_missing_positions.append(position)
            all_label_probs.append(label_probs)
        all_label_probs = np.array(all_label_probs) # prob not normalized

        return all_label_probs # NOT NORMALIZED


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
