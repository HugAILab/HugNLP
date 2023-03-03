# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 9:13 下午
# @Author  : JianingWang
# @File    : cmrc2018
import json
import os.path
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerBase
from processors.ProcessorBase import DataProcessor, CLSProcessor
from collections import defaultdict
from metrics import datatype2metrics


class CMRCProcessor(CLSProcessor):
    def __init__(self, data_args, training_args, model_args, post_tokenizer=False, keep_raw_data=True):
        super().__init__(data_args, training_args, model_args, post_tokenizer=post_tokenizer, keep_raw_data=keep_raw_data)
        self.train_file = os.path.join(data_args.data_dir, "train.json")
        self.dev_file = os.path.join(data_args.data_dir, "dev.json")
        self.test_file = os.path.join(data_args.data_dir, "test.json")
        self.trial_file = os.path.join(data_args.data_dir, "trial.json")
        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.sentence1_key = "content"
        self.labels = [0, 1, 2, 3]

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForTokenClassification(self.tokenizer, padding="longest", pad_to_multiple_of=8 if pad_to_multiple_of_8 else None)

    def get_examples(self, set_type):
        if set_type == "train":
            examples = self._create_examples(self._read_json(self.train_file)["data"] + self._read_json(self.trial_file)["data"], "train")
            # examples = self._create_examples(self._read_json(self.train_file)["data"], "train")
            examples = examples[:self.data_args.max_train_samples]
            self.train_examples = examples
        elif set_type == "dev":
            examples = self._create_examples(self._read_json(self.dev_file)["data"], "dev")
            examples = examples[:self.data_args.max_eval_samples]
            self.dev_examples = examples
        elif set_type == "test":
            examples = self._create_examples(self._read_json(self.test_file)["data"], "test")
            examples = examples[:self.data_args.max_predict_samples]
            self.test_examples = examples
        return examples

    def _create_examples(self, lines, set_type):
        examples = []
        for line in lines:
            for paragraph in line["paragraphs"]:
                context = paragraph["context"].strip()
                for qa in paragraph["qas"]:
                    question = qa["question"].strip()
                    id_ = qa["id"]
                    answers = qa["answers"]
                    if set_type == "train":
                        # assert len(answers) == 1
                        examples.extend(self.stride_split(id_, question, context, answers[0]["text"], answers[0]["answer_start"], is_train=1))
                    elif set_type == "dev":
                        answer_starts = [answer["answer_start"] for answer in answers]
                        answer_text = [answer["text"] for answer in answers]
                        o = self.stride_split(id_, question, context, answer_text[0], answer_starts[0], is_train=1)
                        for i in o:
                            i["answer_all"] = answer_text
                        examples.extend(o)
                    else:
                        examples.extend(self.stride_split(id_, question, context, "", -1))

        return examples

    def stride_split(self, i, q, c, a, s, is_train=0):
        """滑动窗口分割context
        """
        # 标准转换
        # q = lowercase_and_normalize(q)
        # c = lowercase_and_normalize(c)
        # b = lowercase_and_normalize(a)
        if a and a[0] == " ":
            a = a[1:]
            s = s + 1
        if a and a[-1] == " ":
            a = a[:-1]
        if a in ["中医认为鲈鱼性温味甘，有健脾胃、补肝肾、止咳化痰的作用。", "北京故宫博物院"]:
            s = s - 1

        e = s + len(a)
        # 滑窗分割
        results, n = [], 0
        max_c_len = self.max_len - len(q) - 3
        while True:
            l, r = n * self.doc_stride, n * self.doc_stride + max_c_len
            if l <= s < e <= r:
                results.append({"id": i, "question": q, "content": c[l:r], "answer": a, "start": s - l, "end": e - l, "is_train": is_train})
            else:
                results.append({"id": i, "question": q, "content": c[l:r], "answer": "", "start": -1, "end": -1, "is_train": is_train})
            if r >= len(c):
                return results
            n += 1

    def build_preprocess_function(self):
        # Tokenize the texts
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length

        def func(examples):
            # Tokenize
            starts, ends, answers = examples["start"], examples["end"], examples["answer"]
            is_train = max(starts) > 0
            sep_id = tokenizer.sep_token_id
            tokenized_examples = tokenizer(
                examples["question"],
                examples["content"],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length" if self.data_args.pad_to_max_length else False,
                return_offsets_mapping=True
            )
            # 确定label
            offset_mapping = tokenized_examples["offset_mapping"]
            labels = []
            if is_train:
                for input_id, offset, start, end, answer in zip(tokenized_examples["input_ids"], offset_mapping, starts, ends, answers):
                    label = [0] * len(input_id)
                    sep_idx_1 = input_id.index(sep_id)
                    for i, (m, n) in enumerate(offset):
                        if i <= sep_idx_1:
                            continue
                        if m <= start < n:
                            label[i] = 1
                        if m <= end - 1 < n:
                            if label[i] == 1:
                                label[i] = 3
                            else:
                                label[i] = 2
                    w = [i for i, j in enumerate(label) if j > 0]
                    # if len(w) not in [0, 2]:
                    #     t = tokenizer.decode(input_id[w[0]])
                    #     if t != "[UNK]":
                    #         assert t == answer.lower()
                    # p = tokenizer.decode(input_id[w[0]:w[1]]).replace(" ", "")
                    # if p != answer:
                    #     print(p, answer)
                    labels.append(label)
            else:
                for i in range(len(examples["question"])):
                    labels.append([0] * len(tokenized_examples["input_ids"][i]))

            tokenized_examples["labels"] = labels
            return tokenized_examples

        return func

    def get_predict_answer(self, dataset_name, predictions, label_ids):
        max_answer_len = 100
        from scipy.special import softmax
        probs = softmax(predictions, axis=2)

        examples = self.raw_datasets[dataset_name]
        example_id_feature_map = defaultdict(list)
        for feature_id, example in enumerate(examples):
            example_id_feature_map[example["id"]].append(feature_id)
        if label_ids is not None:
            probs[label_ids == -100, :] = 0

        start_prob = probs[:, :, 1]
        end_prob = probs[:, :, 2]
        num_sample, max_seq_len = start_prob.shape
        w = range(num_sample)
        start_idx = np.full(num_sample, max_seq_len - 2)
        end_idx = np.full(num_sample, max_seq_len - 1)
        max_end = np.full(num_sample, max_seq_len - 1)
        max_sum = start_prob[w, start_idx] + end_prob[w, end_idx]
        for i in range(max_seq_len - 3, -1, -1):
            max_end[end_prob[w, i + 1] > end_prob[w, max_end]] = i + 1
            s = start_prob[w, i] + end_prob[w, max_end]
            k = s > max_sum
            max_sum[k] = s[k]
            start_idx[k] = i
            end_idx[k] = max_end[k]
        feature_map_prob = max_sum
        feature_answer = np.concatenate((start_idx.reshape(-1, 1), end_idx.reshape(-1, 1)), axis=1)

        # 单个token
        single_sum_prob = probs[:, :, 3] * 2
        max_start = np.argmax(single_sum_prob, axis=1)
        max_end = max_start
        max_pos = np.concatenate((max_start.reshape(-1, 1), max_end.reshape(-1, 1)), axis=1)
        max_prob = single_sum_prob.max(axis=1)
        new = max_prob > feature_map_prob
        feature_map_prob[new] = max_prob[new]
        feature_answer[new] = max_pos[new]

        predict_answers = {}
        for example_id, feature_ids in example_id_feature_map.items():
            best_feature = np.argmax(feature_map_prob[feature_ids])
            best_start_end = feature_answer[feature_ids][best_feature]
            best_example = examples[feature_ids[best_feature]]
            assert best_example["id"] == example_id
            if best_start_end[0] > len(best_example["offset_mapping"]):
                print(1)
            s = best_example["offset_mapping"][best_start_end[0]][0]
            e = best_example["offset_mapping"][best_start_end[1]][1]
            answer = best_example["content"][s: e]
            predict_answers[example_id] = answer
        return predict_answers

    def compute_metrics(self, eval_predictions):
        from processors.benchmark.cluemrc.cmrc_evaluate import evaluate2
        predictions, label_ids = eval_predictions
        if label_ids.max() == 0:
            return {"eval_em": 0, "eval_f1": 0}
        predict_answers, _ = self.get_predict_answer("validation", predictions, label_ids)
        # predict_answers2, _ = self.get_predict_answer2("validation", predictions, label_ids)
        true_labels = {}
        for example in self.raw_datasets["validation"]:
            if example["id"] not in true_labels:
                true_labels[example["id"]] = example["answer_all"]
        return evaluate2(predict_answers, true_labels)

    def save_result(self, logits, label_ids):
        predict_answers, top_answers = self.get_predict_answer("test", logits, label_ids)
        outfile = os.path.join(self.training_args.output_dir, "cmrc2018_predict.json")
        with open(outfile, "w", encoding="utf8") as f:
            json.dump(predict_answers, f, ensure_ascii=False, indent=4)


@dataclass
class DataCollatorForGlobalPointer:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    pad_to_max_length: Optional[bool] = None

    def __call__(self, features):
        # Tokenize
        is_train = max([feature["is_train"] for feature in features]) > 0
        batch = []
        for f in features:
            batch.append({"input_ids": f["input_ids"], "token_type_ids": f["token_type_ids"], "attention_mask": f["attention_mask"]})
        batch = self.tokenizer.pad(
            batch,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        # 确定label
        if not is_train:
            return batch
        else:
            labels = torch.zeros(len(features), 1, self.max_length, self.max_length)  # 阅读理解任务entity种类为1
            for feature_id, feature in enumerate(features):
                start, end = feature["start"], feature["end"]
                offset = feature["offset_mapping"]
                position_map = {}
                # sep_id = batch["input_ids"][feature_id].index(self.tokenizer.sep_token_id)
                for i, (m, n) in enumerate(offset):
                    # if i < sep_id:
                    #     continue
                    for k in range(m, n + 1):
                        position_map[k] = i
                if start > 0 and end > 0:

                    labels[feature_id, 0, position_map[start], position_map[end] - 1] = 1

                else:
                    # 不包含答案时则将label指向[CLS]
                    labels[feature_id, 0, 0, 0] = 1
            batch["labels"] = labels
            if batch["labels"].max() > 0:
                batch["short_labels"] = torch.ones(len(features))
            else:
                batch["short_labels"] = torch.zeros(len(features))
            return batch


class CMRCGPProcessor(CMRCProcessor):
    def __init__(self, data_args, training_args, model_args):
        super().__init__(data_args, training_args, model_args, keep_raw_data=True)
        self.sentence1_key = None
        self.example_id_feature_map = {}
        self.offset_map = {}
        self.content_map = {}

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForGlobalPointer(self.tokenizer, pad_to_multiple_of=8 if pad_to_multiple_of_8 else None, pad_to_max_length=self.data_args.pad_to_max_length)

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
                examples["question"],
                examples["content"],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length" if self.data_args.pad_to_max_length else False,
                return_offsets_mapping=True
            )
            # 确定label
            return tokenized_examples

        return func

    def get_predict_answer(self, dataset_name, predictions, label_ids, return_top=False):
        probs, indices = predictions
        probs = probs.squeeze(1)
        indices = indices.squeeze(1)
        examples = self.raw_datasets[dataset_name]
        assert dataset_name in ["validation", "test"]
        if dataset_name not in self.example_id_feature_map:
            self.example_id_feature_map[dataset_name] = defaultdict(list)
            for feature_id, example in enumerate(examples):
                self.example_id_feature_map[dataset_name][example["id"]].append(feature_id)
            self.offset_map[dataset_name] = examples["offset_mapping"]
            self.content_map[dataset_name] = examples["content"]
        predict_answers, topk_result = {}, {}
        for example_id, feature_ids in self.example_id_feature_map[dataset_name].items():
            feature_ids = [id_ for id_ in feature_ids if indices[id_][0] != 0]
            if len(feature_ids) == 0:
                predict_answers[example_id] = ""
                topk_result[example_id] = ""
                continue
            feature_scores, feature_indices = probs[feature_ids].flatten().tolist(), indices[feature_ids].flatten().tolist()
            feature_id_flatten = []
            for id_ in feature_ids:
                feature_id_flatten += [id_] * len(probs[feature_ids[0]])
            result = list(zip(feature_scores, feature_indices, feature_id_flatten))

            result = [i for i in result if i[1] != 0]
            best_score, best_start_end, best_feature_id = result[0]
            best_start_end = np.unravel_index(best_start_end, (self.data_args.max_seq_length, self.data_args.max_seq_length))

            # assert best_example["id"] == example_id
            s = self.offset_map[dataset_name][best_feature_id][best_start_end[0]][0]
            e = self.offset_map[dataset_name][best_feature_id][best_start_end[1]][1]
            answer = self.content_map[dataset_name][best_feature_id][s: e]

            predict_answers[example_id] = answer
            if return_top:
                out = []
                for prob, se, fid in result:
                    se = np.unravel_index(se, (self.data_args.max_seq_length, self.data_args.max_seq_length))
                    s = self.offset_map[dataset_name][fid][se[0]][0]
                    e = self.offset_map[dataset_name][fid][se[1]][1]
                    answer = self.content_map[dataset_name][fid][s: e]
                    out.append({"prob": prob, "answer": answer})
                    # out.append({"prob": prob, "answer": answer, "fid": fid, "start_end": [int(se[0]), int(se[1])]})
                topk_result[example_id] = out[:20]
        return predict_answers, topk_result

    # def compute_metrics(self, eval_predictions):
    #     examples = self.raw_datasets["validation"]
    #     golden, dataname_map, dataname_type = {}, defaultdict(list), {}
    #     predictions, _ = self.get_predict_answer("validation", eval_predictions[0], eval_predictions[1], True)
    #     for example in examples:
    #         id_ = example["id"]
    #         golden[id_] = example["target"]
    #
    #     all_metrics = {
    #         "macro_f1": 0.,
    #         "micro_f1": 0.,
    #         "eval_num": 0,
    #     }
    #
    #     metric = datatype2metrics["mrc"]()
    #     gold = {k: v for k, v in golden.items()}
    #     pred = {k: v for k, v in predictions.items()}
    #     # pred = {"dev-{}".format(value["id"]): value["label"] for value in predictions if "dev-{}".format(value["id"]) in data_ids}
    #     score = metric.calc_metric(golden=gold, predictions=pred)
    #     acc, f1 = score["acc"], score["f1"]
    #     all_metrics["macro_f1"] += f1
    #     return all_metrics

    def save_result(self, logits, label_ids):
        predict_answers, top_answers = self.get_predict_answer("test", logits, label_ids, return_top=True)
        outfile = os.path.join(self.training_args.output_dir, "cmrc2018_predict.json")
        with open(outfile, "w", encoding="utf8") as f:
            json.dump(predict_answers, f, ensure_ascii=False, indent=4)
        topfile = os.path.join(self.training_args.output_dir, "top20_predict.json")
        with open(topfile, "w", encoding="utf8") as f2:
            json.dump(top_answers, f2, ensure_ascii=False, indent=4)
