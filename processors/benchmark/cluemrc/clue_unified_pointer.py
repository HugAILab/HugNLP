# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 9:13 下午
# @Author  : JianingWang
# @File    : clue_unified_pointer
import json
import torch
import os.path
import numpy as np
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
from transformers import PreTrainedTokenizerBase
from processors.ProcessorBase import CLSProcessor
from metrics import datatype2metrics

# CLUE榜单，除了多项选择外，其余任务全部转换为MRC任务

@dataclass
class DataCollatorForGlobalPointer:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    pad_to_max_length: Optional[bool] = None

    def __call__(self, features):
        # Tokenize
        is_train = features[0]['is_train'] > 0
        batch = []
        for f in features:
            batch.append({'input_ids': f['input_ids'],
                          'token_type_ids': f['token_type_ids'],
                          'attention_mask': f['attention_mask']})
        batch = self.tokenizer.pad(
            batch,
            padding='max_length',  # 为了index不出错直接Padding到max length，如果用longest，后面的np.unravel_index也要改
            max_length=self.max_length,
            return_tensors="pt"
        )
        # 确定label
        if not is_train:
            return batch
        else:
            # label之所以这样设置，是为了适应于多区间阅读理解任务（多标签分类）
            labels = torch.zeros(len(features), 1, self.max_length, self.max_length)  # 阅读理解任务entity种类为1 [bz, 1, max_len, max_len]
            for feature_id, feature in enumerate(features): # 遍历每个样本
                starts, ends = feature['start'], feature['end']
                offset = feature['offset_mapping'] # 表示tokenizer生成的token对应原始文本中字符级别的位置区间
                position_map = {}
                for i, (m, n) in enumerate(offset):
                    if i != 0 and m == 0 and n == 0:
                        continue
                    for k in range(m, n + 1):
                        position_map[k] = i # 字符级别的第k个字符属于分词i
                for start, end in zip(starts, ends):
                    end -= 1
                    # MRC 没有答案时则把label指向CLS
                    if start == 0:
                        assert end == -1
                        labels[feature_id, 0, 0, 0] = 1
                    else:
                        if start in position_map and end in position_map:
                            # 指定下列元素为1，说明表示第feature_id个样本的预测区间
                            labels[feature_id, 0, position_map[start], position_map[end]] = 1

            # short_labels没用，解决transformers trainer默认会merge labels导致内存爆炸的问题
            # 需配合--label_names=short_labels使用
            batch['labels'] = labels
            if batch['labels'].max() > 0:
                batch['short_labels'] = torch.ones(len(features))
            else:
                batch['short_labels'] = torch.zeros(len(features))
            return batch


class CPICProcessor(CLSProcessor):
    def __init__(self, data_args, training_args, model_args, post_tokenizer=False, keep_raw_data=True):
        super().__init__(data_args, training_args, model_args, post_tokenizer=post_tokenizer, keep_raw_data=keep_raw_data)
        self.train_file = os.path.join(data_args.data_dir, 'train.json')
        self.dev_file = os.path.join(data_args.data_dir, 'dev.json')
        self.test_file = os.path.join(data_args.data_dir, 'test.json')
        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.sentence1_key = None

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForGlobalPointer(self.tokenizer, pad_to_multiple_of=8 if pad_to_multiple_of_8 else None, pad_to_max_length=self.data_args.pad_to_max_length)

    def get_examples(self, set_type):
        if set_type == 'train':
            examples = self._create_examples(self._read_json(self.train_file), 'train')
            # 使用 open data + 比赛训练数据直接训练
            # examples = self._create_examples(self._read_json(self.train_file) + self._read_json(self.dev_file) * 2, 'train')
            examples = examples[:self.data_args.max_train_samples]
            self.train_examples = examples
        elif set_type == 'dev':
            examples = self._create_examples(self._read_json(self.dev_file), 'dev')
            examples = examples[:self.data_args.max_eval_samples]
            self.dev_examples = examples
        elif set_type == 'test':
            examples = self._create_examples(self._read_json(self.test_file), 'test')
            examples = examples[:self.data_args.max_predict_samples]
            self.test_examples = examples
        return examples

    def _create_examples(self, lines, set_type):
        examples = []
        is_train = 0 if set_type == 'test' else 1
        for line in lines:
            id_ = line['ID'] # 原始数据的编号
            text = line['instruction'] # 原始文本+候选+模板形成的最终输入序列
            target = line['target'] # 目标答案
            start = line['start'] # 目标答案在输入序列的起始位置
            data_type = line['data_type'] # 该任务的类型
            if data_type == 'ner':
                new_start, new_end = [], []
                for t, entity_starts in zip(target, start):
                    for s in entity_starts:
                        new_start.append(s)
                        new_end.append(s + len(t))
                start, end = new_start, new_end
                target = '|'.join(target)
            else:
                start, end = [start], [start + len(target)]

            examples.append({'id': id_,
                             'content': text,
                             'start': start,
                             'end': end,
                             'target': target,
                             'data_type': data_type,
                             'is_train': is_train})

        return examples

    def set_config(self, config):
        config.ent_type_size = 1
        config.inner_dim = 64
        config.RoPE = True

    def build_preprocess_function(self):
        # Tokenize the texts
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length

        def func(examples):
            # Tokenize
            tokenized_examples = tokenizer(
                examples['content'],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length" if self.data_args.pad_to_max_length else False,
                return_offsets_mapping=True
            )
            # 确定label
            return tokenized_examples

        return func

    def get_predict_result(self, logits, examples):
        probs, indices = logits
        probs = probs.squeeze(1)  # topk结果的概率
        indices = indices.squeeze(1)  # topk结果的索引
        predictions = {}
        for prob, index, example in zip(probs, indices, examples):
            data_type = example['data_type']
            id_ = example['id']
            if data_type == 'ner':
                answer = []
                # TODO 1. 调节阈值 2. 处理输出实体重叠问题
                entity_index = index[prob > 0.6]
                for entity in entity_index:
                    # 1D index转2D index
                    start_end = np.unravel_index(entity, (self.data_args.max_seq_length, self.data_args.max_seq_length))
                    s = example['offset_mapping'][start_end[0]][0]
                    e = example['offset_mapping'][start_end[1]][1]
                    ans = example['content'][s: e]
                    if ans not in answer:
                        answer.append(ans)
                predictions[id_] = answer
            else:
                best_start_end = np.unravel_index(index[0], (self.data_args.max_seq_length, self.data_args.max_seq_length))
                s = example['offset_mapping'][best_start_end[0]][0]
                e = example['offset_mapping'][best_start_end[1]][1]
                answer = example['content'][s: e]
                predictions[id_] = answer
        return predictions

    def compute_metrics(self, eval_predictions):
        examples = self.raw_datasets['validation']
        golden, dataname_map, dataname_type = {}, defaultdict(list), {}
        predictions = self.get_predict_result(eval_predictions[0], examples)
        for example in examples:
            data_type = example['data_type']
            dataname = "_".join(example["id"].split("_")[:-1])
            if dataname not in dataname_type:
                dataname_type[dataname] = data_type
            id_ = example['id']
            dataname_map[dataname].append(id_)
            if data_type == 'ner':
                golden[id_] = example['target'].split('|')
            else:
                golden[id_] = example['target']

        all_metrics = {
            "macro_f1": 0.,
            "micro_f1": 0.,
            "eval_num": 0,
        }

        for dataname, data_ids in dataname_map.items():
            metric = datatype2metrics[dataname_type[dataname]]()
            gold = {k: v for k, v in golden.items() if k in data_ids}
            pred = {k: v for k, v in predictions.items() if k in data_ids}
            score = metric.calc_metric(golden=gold, predictions=pred)
            acc, f1 = score['acc'], score['f1']
            if len(gold) != len(pred) or len(gold) < 20:
                print(dataname, dataname_type[dataname], round(acc, 4), len(gold), len(pred), data_ids)
            all_metrics["macro_f1"] += f1
            all_metrics["micro_f1"] += f1 * len(data_ids)
            all_metrics["eval_num"] += len(data_ids)
            all_metrics[dataname] = round(acc, 4)
        all_metrics["macro_f1"] = round(all_metrics["macro_f1"] / len(dataname_map), 4)
        all_metrics["micro_f1"] = round(all_metrics["micro_f1"] / all_metrics["eval_num"], 4)
        return all_metrics

    def save_result(self, logits, label_ids):
        examples = self.raw_datasets['test']
        predicts = self.get_predict_result(logits, examples)

        outfile = os.path.join(self.training_args.output_dir, 'answer.json')
        with open(outfile, 'w', encoding='utf8') as f:
            json.dump(predicts, f, ensure_ascii=False, indent=2)
