# -*- coding: utf-8 -*-
# @Time    : 2022/2/27 12:58 下午
# @Author  : JianingWang
# @File    : c3
import json
import os.path
from dataclasses import dataclass
from itertools import chain
from typing import Union, Optional
import numpy as np
import torch

from processors.ProcessorBase import DataProcessor, CLSProcessor
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 512
    eval_max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    is_duma: Optional[bool] = False
    pseudo_train: Optional[bool] = False

    def __call__(self, features):
        first_sentences, second_sentences, labels = [], [], []
        for feature in features:
            first_sentences.extend([feature['text_a']] * 4)
            header = feature['question']
            for i, choice in enumerate(feature['choices']):
                second_sentences.append(f"{header} {choice}")
            if 'label' in feature and feature['label'] is not None:
                labels.append(feature['label'])
        length = self.max_length if labels else self.eval_max_length
        # Tokenize
        batch = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=length,
            padding='longest',
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        if self.is_duma:
            pq_end_pos = batch['input_ids'] == self.tokenizer.sep_token_id
            pq_end_pos = pq_end_pos.nonzero()[:, 1].view(-1, 2)
            pq_end_pos -= 1
            batch['pq_end_pos'] = pq_end_pos

        batch_size = len(features)
        num_choices = len(features[0]['choices'])
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        if not labels:
            return batch
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        if self.pseudo_train:
            pseudo = [feature.pop('pseudo') for feature in features]
            batch['pseudo'] = torch.tensor(pseudo, dtype=torch.int64)
        # batch['token_type_ids'] = torch.zeros_like(batch['input_ids'], dtype=torch.long)
        return batch


class C3Processor(CLSProcessor):
    def __init__(self, data_args, training_args, model_args):
        super().__init__(data_args, training_args, model_args, post_tokenizer=True)
        self.mtrain_file = os.path.join(data_args.data_dir, 'm-train.json')
        self.dtrain_file = os.path.join(data_args.data_dir, 'd-train.json')
        self.mdev_file = os.path.join(data_args.data_dir, 'm-dev.json')
        self.ddev_file = os.path.join(data_args.data_dir, 'd-dev.json')
        self.test_file = os.path.join(data_args.data_dir, 'test1.1.json')
        self.pseudo_train = False

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        is_duma = self.data_args.task_type.startswith('duma')
        eval_max_length = self.data_args.max_eval_seq_length if self.data_args.max_eval_seq_length else self.data_args.max_seq_length
        return DataCollatorForMultipleChoice(self.tokenizer,
                                             max_length=self.data_args.max_seq_length,
                                             eval_max_length=eval_max_length,
                                             pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
                                             is_duma=is_duma,
                                             pseudo_train=self.pseudo_train)

    def get_examples(self, set_type):
        if set_type == 'train':
            examples = self._create_examples(self._read_json(self.mtrain_file) + self._read_json(self.dtrain_file)*2, 'train')
            self.train_examples = examples
            if self.pseudo_train:
                predict_file = os.path.join(self.data_args.data_dir, 'c3_predict.json')
                pse1 = self._create_examples(self._read_json(self.mdev_file)*2 + self._read_json(self.ddev_file)*2, 'train')
                pse2 = self._create_examples(self._read_json(predict_file), 'train', pseudo=True)
                examples = examples + pse1 + pse2
        elif set_type == 'dev':
            examples = self._create_examples(self._read_json(self.mdev_file) + self._read_json(self.ddev_file), 'dev')
            self.dev_examples = examples
        elif set_type == 'test':
            examples = self._create_examples(self._read_json(self.test_file), 'test')
            self.test_examples = examples
        return examples

    def _create_examples(self, lines, set_type, pseudo=False):
        num_map = {'①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5, '⑥': 6}
        examples = []
        for line in lines:
            content, qas = line[0], line[1]
            # if set_type != 'test':
            #     eid = line[2]
            eid = None
            text = ''.join(content)
            for qa in qas:
                q = qa['question']
                choices = qa['choice']
                if set_type == 'test':
                    eid = qa['id']
                if len(choices) > 4:
                    choices_convert = []
                    for choice in choices[-4:]:
                        if choice[0] not in num_map:
                            choices_convert = choices[:4]
                            continue
                        choices_convert.append(''.join([choices[num_map[n] - 1] for n in choice]))
                    choices = choices_convert
                for _ in range(len(choices), 4):
                    choices.append('无效答案')
                try:
                    label = choices.index(qa['answer']) if 'answer' in qa else None
                    text_a = self.cut(text, q, choices)
                    examples.append({'text_a': text_a,
                                     'question': q,
                                     'choices': choices,
                                     'label': label,
                                     'eid': eid,
                                     'pseudo': 1 if pseudo else 0})
                except:
                    logger.error('load data error q: {}, choices: {}'.format(q, choices))
                    pass
        return examples

    def cut(self, text, q, choices):
        from math import ceil
        try:
            max_len = self.data_args.max_seq_length - len(q) - max([len(i) for i in choices if i != '无效答案']) - 4
        except:
            print(text, q, choices)
        t_len = len(text)
        window = 64
        assert window < max_len
        if t_len < max_len:
            return text
        text_split = []
        n = ceil((t_len - max_len) / window) + 1
        for i in range(0, n):
            if i == n - 1:
                text_split.append(text[-max_len:])
            else:
                text_split.append(text[window * i: window * i + max_len])

        qa = q + ''.join([i for i in choices if i != '无效答案'])
        scores = [len(set(qa) & set(i)) for i in text_split]
        max_idx = scores.index(max(scores))
        return text_split[max_idx]

    def get_predict_result(self, logits, examples):
        pass

    def compute_metrics(self, eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        res = {"eval_acc": (preds == label_ids).astype(np.float32).mean().item()}
        # print('res=', res)
        return res

    def save_result(self, logits, label_ids):
        predictions = np.argmax(logits, axis=1)
        predicts = [{"id": k, "label": int(v)} for k, v in zip([i['eid'] for i in self.test_examples], predictions)]
        outfile = os.path.join(self.training_args.output_dir, 'c311_predict.json')
        with open(outfile, 'w', encoding='utf8') as f:
            for p in predicts:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')
