# -*- coding: utf-8 -*-
# @Time    : 2023/3/5 11:10 下午
# @Author  : JianingWang
# @File    : generative_evaluator.py

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
from tools.runner_utils.log_util import logging
from tools.computations.softmax import softmax
from metrics.generation_metric import GenerationMetric

logger = logging.getLogger(__name__)



"""
Evaluator for the task of generation with Encoder-Decoder PLMs.
"""
class EncoderDecoderGenerationEvaluator(GenerationEvaluator):
    # TODO
    pass

"""
Evaluator for the task of generation with Causal PLMs.
"""
class CausalGenerationEvaluator(GenerationEvaluator):

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
        assert hasattr(self.processor, "input_key"), "If you choose generation task, you must define 'input_key' in the processor."
        assert hasattr(self.processor, "output_key"), "If you choose generation task, you must define 'output_key' in the processor."
        self.input_key = self.processor.input_key # e.g. 'input'
        self.output_key = self.processor.output_key # e.g. 'output'

    def default_compute_metrics(self, predictions, examples):
        """
        predictions: [[["x", "x", ...], ...], ...]
        examples[0]["output"]: [["x", "x", ...], ...]
        """

        golden = {}
        # predictions:  {"xx": "xxx", ...}
        for idx, example in enumerate(examples):
            label = example[self.output_key]
            if type(label) == str:
                label_tokens = self.processor.tokenizer.tokenize(label)
            golden[idx] = label_tokens

        all_metrics = {
            "em": 0.,
            "bleu-1": 0.,
            "bleu-2": 0.,
            "bleu-3": 0.,
            "bleu-4": 0.,
        }

        metric = GenerationMetric()
        gold = {k: v for k, v in golden.items()}
        pred = {k: v for k, v in enumerate(predictions)}
        score = metric.calc_metric(golden=gold, predictions=pred)
        em, bleu1, bleu2, bleu3, bleu4 = score["em"], score["bleu-1"], score["bleu-2"], score["bleu-3"], score["bleu-4"]
        all_metrics["em"] += em
        all_metrics["bleu-1"] += bleu1
        all_metrics["bleu-2"] += bleu2
        all_metrics["bleu-3"] += bleu3
        all_metrics["bleu-4"] += bleu4
        return all_metrics


    def evaluate(self):

        # generate answer for validation set
        all_raw_answers, all_generated_texts = self.generate(
            eval_dataset=self.eval_dataset,
            num_log_probs=100,
            echo=False, # echo is False means do not directly obtain label probability.
        )

        metrics = self.default_compute_metrics(all_generated_texts, self.eval_dataset)
        # print("dev metrics=", metrics)

        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)


    def predict(self):

        # generate answer for test dataset
        all_raw_answers, all_generated_texts = self.generate(
            eval_dataset=self.test_dataset,
            num_log_probs=100,
            echo=False
        )

        if self.output_key in self.test_dataset.features and not self.data_args.keep_predict_labels:

            metrics = self.default_compute_metrics(all_generated_texts, self.test_dataset)
            # print("test metrics=", metrics)

            self.trainer.log_metrics("test", metrics)
            self.trainer.save_metrics("test", metrics)

        # If you have defined save_result function in the processor.
        if hasattr(self.processor, "save_result"):
            assert self.paradigm == NO_GENERATE, "default processor only support no-generate model."
            if self.trainer.is_world_process_zero():
                if self.output_key not in self.test_dataset.features:
                    self.processor.save_result(all_generated_texts)
                else:
                    self.processor.save_result(all_generated_texts, self.test_dataset[self.output_key])
        else:
            # If you not define the save_result function.
            examples = self.test_dataset
            # submit
            answer = list()
            for k, token_lists in enumerate(all_generated_texts):
                input_text = examples[k][self.input_key]
                res = list()
                for token_list in token_lists:
                    res.append(self.processor.tokenizer.decode(self.processor.tokenizer.encode(token_list, add_special_tokens=False)))
                answer.append({"id": k, "input": input_text, "output": res})

            output_submit_file = os.path.join(self.training_args.output_dir, "answer.json")
            # Save the label results
            with open(output_submit_file, "w") as writer:
                for i, pred in enumerate(answer):
                    json_d = {}
                    json_d["id"] = i
                    json_d["input"] = pred["input"]
                    json_d["output"] = pred["output"]
                    writer.write(json.dumps(json_d) + "\n")


    def generate(self, eval_dataset, num_log_probs=100, echo=False):
        # obtain generative answer from causal PLM.
        assert hasattr(self.processor, "l"), "You must define l ('max length of generated tokens') for generative-style tasks"
        l = self.processor.l
        all_raw_answers = []
        all_generated_texts = []
        for ei, data in enumerate(tqdm(eval_dataset)):
            total_sequences = self.model.generate(
                input_ids=torch.Tensor([data['input_ids']]).long().to(self.model.device),
                attention_mask=torch.Tensor([data['attention_mask']]).long().to(self.model.device),
                max_length=l + len(data['input_ids']),
                do_sample=False,
                temperature=0.8,
                top_p=0.95,
                num_beams=5,
                top_k=5,
                num_return_sequences=5,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            if ei < 5:
                # display a few generated result. e.g. ['Ġnegative']
                print("display generated result #{}={}".format(ei, self.processor.tokenizer.convert_ids_to_tokens(total_sequences[0][len(data['input_ids']):])))
            if num_log_probs is not None:
                return_raw_answers = list()
                return_generation_texts = list()
                # we are left padding, so we need to adjust the position IDs
                attention_mask = (total_sequences != 50256).float()
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                # get the logits for the context and the next l tokens
                logits = self.model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu()
                response_res = self.generation_model_response.call_for_gpt2_response(self.processor.tokenizer, logits, total_sequences, l, num_log_probs, echo)

                for answer_id, answer in enumerate(response_res['choices']):
                    return_raw_answers.append(answer)
                    return_generation_texts.append(answer["tokens"])
                all_raw_answers.append(return_raw_answers)
                all_generated_texts.append(return_generation_texts)

        """
        all_raw_answers:
        [
            [
                {
                    'text': ' I', # the top 1 generated token
                    'logprobs': {'top_logprobs': [{' I': -3.4267239570617676, '\n': -3.5073862075805664, ...], # Top k tokens at this position
                    'token_logprobs': [-3.4267239570617676], # the top 1 generated token score
                    'tokens': [' I']} # the generated token list
                },
                ...
            ],
            ...
        ]
        """
        return all_raw_answers, all_generated_texts


    def obtain_label_logits(self, answers, examples):
        pass


    def get_best_and_topk(self, logits, examples, topk=10, stage="dev"):
        pass
