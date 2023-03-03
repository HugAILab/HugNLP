# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 10:19 下午
# @Author  : JianingWang
# @File    : data_process.py
import json
import torch
import os.path
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

from processors.ProcessorBase import DataProcessor, FewShotNERProcessor
from metrics import datatype2metrics
from collections import defaultdict, Counter
from torch import distributed as dist

from datasets import DatasetDict
from processors.dataset import DatasetK
from processors.ner.fewshot_ner.data_collator import DataCollatorForSpanProto, DataCollatorForTokenProto



"""
Processing data for FewNERD dataset based on token-based proto

"""
class TokenProtoFewNERDProcessor(FewShotNERProcessor):
    def __init__(self, data_args, training_args, model_args, tokenizer=None, post_tokenizer=False, keep_raw_data=True):
        super().__init__(data_args, training_args, model_args, tokenizer, post_tokenizer=post_tokenizer, keep_raw_data=keep_raw_data)
        param = {p.split("=")[0]: p.split("=")[1] for p in (data_args.user_defined).split(" ")} # user_defined parameter
        N, Q, K, mode = param["N"], param["Q"], param["K"], param["mode"] # N: num class, Q: query entity num, K: support entity num
        self.train_file = os.path.join(data_args.data_dir, "train_{}_{}.jsonl".format(N, K))
        self.dev_file = os.path.join(data_args.data_dir, "dev_{}_{}.jsonl".format(N, K))
        self.test_file = os.path.join(data_args.data_dir, "test_{}_{}.jsonl".format(N, K))

        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.sentence1_key = None
        self.mode = mode
        self.num_class = int(N)
        self.num_example = int(K)

        self.ignore_label_id = -1
        self.max_length = self.data_args.max_seq_length

        self.output_dir = "./outputs/{}-{}-{}".format(self.mode, self.num_class, self.num_example)


    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForTokenProto(
            self.tokenizer,
            num_class=self.num_class,
            num_example=self.num_example,
            mode=self.mode,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            pad_to_max_length=self.data_args.pad_to_max_length,
        )

    def __load_data_from_file__(self, filepath):
        with open(filepath)as f:
            lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = json.loads(lines[i].strip())
        return lines

    def get_examples(self, set_type):
        if set_type == "train":
            examples = self._create_examples(self.__load_data_from_file__(self.train_file), set_type)
            self.train_examples = examples
        elif set_type == "dev":
            examples = self._create_examples(self.__load_data_from_file__(self.dev_file), set_type)
            self.dev_examples = examples
        elif set_type == "test":
            examples = self._create_examples(self.__load_data_from_file__(self.test_file), set_type)
            self.test_file = examples
        else:
            examples = None
        return examples

    def get_tokenized_datasets(self):
        raw_datasets = DatasetDict()
        if self.training_args.do_train:
            train_examples = self.get_examples("train") #
            raw_datasets["train"] = DatasetK.from_dict(
                self.list_2_json(train_examples))  # [{k1: xxx, k2: xxx}, {...}] -> {k1: [xxx, xxx], k2: [xxx, xxx]}
        if self.training_args.do_eval:
            dev_examples = self.get_examples("dev")
            raw_datasets["validation"] = DatasetK.from_dict(self.list_2_json(dev_examples))
        if self.training_args.do_predict:
            test_examples = self.get_examples("test")
            raw_datasets["test"] = DatasetK.from_dict(self.list_2_json(test_examples))

        if self.post_tokenizer:
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets
            return raw_datasets
        # datasets的bug, 对于from_dict不会创建cache,需要指定cache_file_names
        # 指定了cache_file_names在_map_single中也需要cache_files不为空才能读取cache
        for key, value in raw_datasets.items():
            value.set_cache_files(["cache_local"])
        # remove_columns = [self.support_key, self.query_key]
        tokenize_func = self.build_preprocess_function()
        # 多gpu, 0计算完存cache，其他load cache
        load_from_cache_file = not self.data_args.overwrite_cache if self.training_args.local_rank in [-1, 0] else True
        base_cache_dir = os.path.join(self.model_args.cache_dir,
                                      "datasets") if self.model_args.cache_dir else os.path.join(
            os.path.expanduser("~"), ".cache/huggingface/datasets/")
        cache_dir = os.path.join(base_cache_dir, self.data_args.task_name)

        os.makedirs(cache_dir, exist_ok=True)
        with self.training_args.main_process_first(desc="dataset tokenizer map"):
            raw_datasets = raw_datasets.map(
                tokenize_func,
                batched=True,
                load_from_cache_file=load_from_cache_file,
                desc="Running tokenizer on dataset",
                cache_file_names={
                    k: f"{cache_dir}/cache_{self.data_args.task_name}_{self.data_args.data_dir.split('/')[-1]}_{self.mode}_{self.num_class}_{self.num_example}_{str(k)}.arrow"
                    for k in raw_datasets},
                num_proc=self.data_args.preprocessing_num_workers,
                # remove_columns=remove_columns
            )
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets
            print("datasets=", raw_datasets)
            return raw_datasets

    def getraw(self, tokens, labels):
        # 分词、获得input_id，attention mask和segment id
        # get tokenized word list, attention mask, text mask (mask [CLS], [SEP] as well), tags

        # split into chunks of length (max_length-2)
        # 2 is for special tokens [CLS] and [SEP]
        tokens_list = []
        labels_list = []
        while len(tokens) > self.max_length - 2:
            tokens_list.append(tokens[:self.max_length-2])
            tokens = tokens[self.max_length-2:]
            labels_list.append(labels[:self.max_length-2])
            labels = labels[self.max_length-2:]
        if tokens:
            tokens_list.append(tokens)
            labels_list.append(labels)

        # add special tokens and get masks
        indexed_tokens_list = []
        mask_list = []
        text_mask_list = []
        for i, tokens in enumerate(tokens_list):
            # token -> ids
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

            # padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens_list.append(indexed_tokens)

            # mask
            mask = np.zeros((self.max_length), dtype=np.int32)
            mask[:len(tokens)] = 1
            mask_list.append(mask)

            # text mask, also mask [CLS] and [SEP]
            text_mask = np.zeros((self.max_length), dtype=np.int32)
            text_mask[1:len(tokens)-1] = 1
            text_mask_list.append(text_mask)

            assert len(labels_list[i]) == len(tokens) - 2, print(labels_list[i], tokens)
        return indexed_tokens_list, mask_list, text_mask_list, labels_list


    def additem(self, d, word, mask, text_mask, label):
        d["word"] += word
        d["mask"] += mask
        d["label"] += label
        d["text_mask"] += text_mask

    def get_token_label_list(self, words, tags):
        tokens = []
        labels = []
        for word, tag in zip(words, tags):
            word_tokens = self.tokenizer.tokenize(word)
            if word_tokens:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                word_labels = [self.tag2label[tag]] + [self.ignore_label_id] * (len(word_tokens) - 1)
                labels.extend(word_labels)
        return tokens, labels

    def populate(self, data, savelabeldic=False):
        """
        populate samples into data dict
        set savelabeldic=True if you want to save label2tag dict
        "word": tokenized word ids
        "mask": attention mask in BERT
        "label": NER labels
        "sentence_num": number of sentences in this set (a batch contains multiple sets)
        "text_mask": 0 for special tokens and paddings, 1 for real text
        """
        dataset = {"word": [], "mask": [], "label":[], "sentence_num":[], "text_mask":[] }
        for i in range(len(data["word"])):
            tokens, labels = self.get_token_label_list(data["word"][i], data["label"][i])
            word, mask, text_mask, label = self.getraw(tokens, labels)
            word = torch.tensor(word).long()
            mask = torch.tensor(mask).long()
            text_mask = torch.tensor(text_mask).long()
            self.additem(dataset, word, mask, text_mask, label)
        dataset["sentence_num"] = [len(dataset["word"])]
        if savelabeldic:
            dataset["label2tag"] = [self.label2tag]
        return dataset



    def _create_examples(self, lines, set_type):
        examples = []
        # is_train = 0 if set_type == "test" else 1
        for id_, line in enumerate(tqdm(lines)): # 遍历每一行（每一行表示一个episode）
            target_classes = line["types"] # 当前episode对应的所有类别 list
            label2id = {v: ei for ei, v in enumerate(target_classes)}
            support = line["support"]
            query = line["query"]

            distinct_tags = ["O"] + target_classes
            self.tag2label = {tag: idx for idx, tag in enumerate(distinct_tags)}
            self.label2tag = {idx: tag for idx, tag in enumerate(distinct_tags)}
            support_set = self.populate(support)
            query_set = self.populate(query, savelabeldic=True)

            examples.append(
                {
                    # "support_set": support_set, # {"word": [], "mask": [], "label":[], "sentence_num":[], "text_mask":[] }
                    # "query_set": query_set, # {"word": [], "mask": [], "label":[], "sentence_num":[], "text_mask":[] }
                    "support_word": support_set["word"],
                    "support_mask": support_set["mask"],
                    "support_label": support_set["label"],
                    "support_sentence_num": support_set["sentence_num"],
                    "support_text_mask": support_set["text_mask"],
                    "query_word": query_set["word"],
                    "query_mask": query_set["mask"],
                    "query_label": query_set["label"],
                    "query_sentence_num": query_set["sentence_num"],
                    "query_text_mask": query_set["text_mask"],
                }
            )

        return examples

    def set_config(self, config):
        config.ent_type_size = 1
        config.inner_dim = 64
        config.RoPE = True
        return config

    def build_preprocess_function(self):
        # Tokenize the texts
        support_key, query_key = self.support_key, self.query_key
        # input_column = self.input_column
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length
        # label_to_id = self.label_to_id
        # label_column = self.label_column

        def func(examples):
            features = examples
            # features = {
            #     "id": examples["id"],
            #     "support_input": list(),
            #     "query_input": list(),
            # }
            # support_inputs, query_inputs = examples[support_key], examples[query_key]
            # for ei, support_input in enumerate(support_inputs):
            #     # 对每个episode，对support和query进行分词
            #     support_input = (support_input, )
            #     query_input = (query_inputs[ei], )
            #     support_result = tokenizer(
            #         *support_input,
            #         padding=False,
            #         max_length=max_seq_length,
            #         truncation="longest_first",
            #         add_special_tokens=True,
            #         return_offsets_mapping=True
            #     )
            #     query_result = tokenizer(
            #         *query_input,
            #         padding=False,
            #         max_length=max_seq_length,
            #         truncation="longest_first",
            #         add_special_tokens=True,
            #         return_offsets_mapping=True
            #     )
            #     features["support_input"].append(support_result)
            #     features["query_input"].append(query_result)
            # features["support_labeled_spans"] = examples["support_labeled_spans"]
            # features["support_labeled_types"] = examples["support_labeled_types"]
            # features["support_sentence_num"] = examples["support_sentence_num"]
            # features["query_labeled_spans"] = examples["query_labeled_spans"]
            # features["query_labeled_types"] = examples["query_labeled_types"]
            # features["query_sentence_num"] = examples["query_sentence_num"]

            return features

        return func

    def fush_multi_answer(self, has_answer, new_answer):
        # 对于某个id测试集，出现多个example时（例如同一个测试样本使用了多个模板而生成了多个example），此时将预测的topk结果进行合并
        # has为已经合并的结果，new为当前新产生的结果，
        # has格式为 {"ans": {"prob": float(prob[index_ids[ei]]), "pos": (s, e)}, ...}
        # new {"ans": {"prob": float(prob[index_ids[ei]]), "pos": (s, e)}, ...}
        # print("has_answer=", has_answer)
        for ans, value in new_answer.items():
            if ans not in has_answer.keys():
                has_answer[ans] = value
            else:
                has_answer[ans]["prob"] += value["prob"]
                has_answer[ans]["pos"].extend(value["pos"])
        return has_answer



    def get_predict_result(self, logits, examples, stage="dev"):
        """
        logits: 表示模型除了loss部分的输出
            query_spans: list = None # e.g. [[[[1, 3], [6, 9]], ...], ...] # 表示每个episode，每个句子预测的所有span
            proto_logits: list = None # e.g. [[[0, 3], ...], ...] # 表示每个episode，每个句子，每个span预测的类别
            topk_probs: torch.FloatTensor = None
            topk_indices: torch.IntTensor = None
        examples: 表示当前的样本，
            {
                    "id": xx
                    "support_input": {
                        "input_ids": [[xxx], ...],
                        "attention_mask": [[xxx], ...],
                        "token_type_ids": [[xxx], ...],
                        "offset_mapping": xxxx
                    },
                    "query_input": {
                        "input_ids": [[xxx], ...],
                        "attention_mask": [[xxx], ...],
                        "token_type_ids": [[xxx], ...],
                        "offset_mapping": xxx
                    },
                    "support_labeled_spans": [[[x, x], ..], ..],
                    "support_labeled_types": [[xx, ..], ..],
                    "support_sentence_num": xx,
                    "query_labeled_spans": [[[x, x], ..], ..],
                    "query_labeled_types": [[xx, ..], ..],
                    "query_sentence_num": xx,
                    "stage": xx,
                }
        """
        # query_spans, proto_logits, _, __ = logits
        word_size = dist.get_world_size()
        results = dict() # 所有episode的query预测的span以及对应的类别
        for i in range(word_size):
            path = os.path.join(self.output_dir, "predict", "{}_predictions_{}.npy".format(stage, i))
            # path = "./outputs2/predict/predictions_{}.npy".format(i)
            assert os.path.exists(path), "unknown path: {}".format(path)
            if os.path.exists(path):
                res = np.load(path, allow_pickle=True)[()]
                # 合并所有device的结果
                for episode_i, value in res.items():
                    results[episode_i] = value

        predictions = dict()

        for example in examples: # 遍历每个episode
            # 当前episode ground truth
            query_labeled_spans = example["query_labeled_spans"]
            query_labeled_types = example["query_labeled_types"]
            query_offset_mapping = example["query_input"]["offset_mapping"]
            id_ = example["id"]



            new_labeled_spans = list()

            # 对于query label，将字符级别的区间转换为分词级别
            for ei in range(len(query_labeled_spans)):  # 遍历每一个句子
                labeled_span = query_labeled_spans[ei]  # list # 当前句子的所有mention span（字符级别）
                offset = query_offset_mapping[ei]  # 表示tokenizer生成的token对应原始文本中字符级别的位置区间
                new_labeled_span = list()  # 当前句子的所有mention span（token级别）
                # starts, ends = feature["start"], feature["end"]
                # print("starts=", starts)
                # print("ends=", ends)
                position_map = {}
                for i, (m, n) in enumerate(offset):  # 第i个分词对应原始文本中字符级别的区间(m, n)
                    if i != 0 and m == 0 and n == 0:
                        continue
                    for k in range(m, n + 1):
                        position_map[k] = i  # 字符级别的第k个字符属于分词i

                for span in labeled_span:  # 遍历每个span
                    start, end = span
                    end -= 1
                    # MRC 没有答案时则把label指向CLS
                    # if start == 0:
                    #     # assert end == -1
                    #     labels[ei, 0, 0, 0] = 1
                    #     new_labeled_span.append([0, 0])

                    if start in position_map and end in position_map:
                        # 指定下列元素为1，说明表示第feature_id个样本的预测区间
                        new_labeled_span.append([position_map[start], position_map[end]])
                new_labeled_spans.append(new_labeled_span)


            # 获得模型预测
            pred_spans, pred_spans_ = results[id_]["spans"], list()
            pred_types, pred_types_ = results[id_]["types"], list()
            # for spans, types in zip(pred_spans, pred_types): # 遍历每一个句子
            #     # 如果预测的label为unlabeled type，则删除（即虽然预测的是实体，但并不在当前episode规定的类里，按照规则则全部视为非实体）
            #     spans_ = list()
            #     types_ = list()
            #     for ei, type in enumerate(types):
            #         if type != self.num_class:
            #             types_.append(type)
            #             spans_.append(spans[ei])
            #     pred_spans_.append(spans_)
            #     pred_types_.append(types_)

            predictions[id_] = {
                "labeled_spans": new_labeled_spans,
                "labeled_types": query_labeled_types,
                "predicted_spans": pred_spans,
                "predicted_types": pred_types
            }

            # print(" === ")
            # print("labeled_spans=", new_labeled_spans)
            # print("query_labeled_types=", query_labeled_types)
            # print("predicted_spans=", pred_spans)
            # print("predicted_types=", pred_types)

        return predictions

    def compute_metrics(self, eval_predictions, stage="dev"):
        """
        eval_predictions: huggingface
        eval_predictions[0]: logits
        eval_predictions[1]: labels
        # print("raw_datasets=", raw_datasets["validation"])
            Dataset({
                features: ["id", "support_labeled_spans", "support_labeled_types", "support_sentence_num", "query_labeled_spans", "query_labeled_types", "query_sentence_num", "stage", "support_input", "query_input"],
                num_rows: 1000
            })
        """
        all_metrics = {
            "span_precision": 0.,
            "span_recall": 0.,
            "eval_span_f1": 0,
            "class_precision": 0.,
            "class_recall": 0.,
            "eval_class_f1": 0,
        }

        examples = self.raw_datasets["validation"] if stage == "dev" else self.raw_datasets["test"]
        # golden, dataname_map, dataname_type = {}, defaultdict(list), {}
        predictions = self.get_predict_result(eval_predictions[0], examples, stage)

        # === 下面使用Few-NERD官方提供的评测方法 ===

        pred_span_cnt = 0  # pred entity cnt
        label_span_cnt = 0  # true label entity cnt
        correct_span_cnt = 0  # correct predicted entity cnt

        pred_class_cnt = 0  # pred entity cnt
        label_class_cnt = 0  # true label entity cnt
        correct_class_cnt = 0  # correct predicted entity cnt

        # 遍历每个episode
        for episode_id, predicts in predictions.items():
            query_labeled_spans = predicts["labeled_spans"]
            query_labeled_types = predicts["labeled_types"]
            pred_span = predicts["predicted_spans"]
            pred_type = predicts["predicted_types"]
            # 遍历每个句子，为每个label生成所有的span e.g. {label:[(start_pos, end_pos), ...]}
            for label_span, label_type, pred_span, pred_type in zip(
                    query_labeled_spans, query_labeled_types, pred_span, pred_type
            ):
                # 用于评价detector检测区间的效果
                label_span_dict = {0: list()}
                pred_span_dict = {0: list()}
                # 用于评价prototype分类的效果
                label_class_dict = dict()
                pred_class_dict = dict()
                # 遍历每个span
                for span, type in zip(label_span, label_type):
                    label_span_dict[0].append((span[0], span[1]))
                    if type not in label_class_dict.keys():
                        label_class_dict[type] = list()
                    label_class_dict[type].append((span[0], span[1]))

                # 遍历每个span，判断其类别，并加入到对应类别的dict中
                for span, type in zip(pred_span, pred_type):
                    pred_span_dict[0].append((span[0], span[1]))
                    if type == self.num_class or span == [0, 0]:
                        continue
                    if type not in pred_class_dict.keys():
                        pred_class_dict[type] = list()
                    pred_class_dict[type].append((span[0], span[1]))

                tmp_pred_span_cnt, tmp_label_span_cnt, correct_span = self.metrics_by_entity(
                    label_span_dict, pred_span_dict
                )

                tmp_pred_class_cnt, tmp_label_class_cnt, correct_class = self.metrics_by_entity(
                    label_class_dict, pred_class_dict
                )
                pred_span_cnt += tmp_pred_span_cnt
                label_span_cnt += tmp_label_span_cnt
                correct_span_cnt += correct_span

                pred_class_cnt += tmp_pred_class_cnt
                label_class_cnt += tmp_label_class_cnt
                correct_class_cnt += correct_class

        span_precision = correct_span_cnt / pred_span_cnt
        span_recall = correct_span_cnt / label_span_cnt
        try:
            span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall)
        except:
            span_f1 = 0.

        class_precision = correct_class_cnt / pred_class_cnt
        class_recall = correct_class_cnt / label_class_cnt
        try:
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall)
        except:
            class_f1 = 0.
        all_metrics["span_precision"], all_metrics["span_recall"], all_metrics["eval_span_f1"] = \
            span_precision, span_recall, span_f1
        all_metrics["class_precision"], all_metrics["class_recall"], all_metrics["eval_class_f1"] = \
            class_precision, class_recall, class_f1
        print("all_metrics=", all_metrics)
        # === 下面用于计算detector的效果 ===
        # 遍历每个episode，获得预测正确的span个数，

        # for example in examples:
        #     data_type = example["data_type"]
        #     dataname = "_".join(example["id"].split("_")[:-1])
        #     if dataname not in dataname_type:
        #         dataname_type[dataname] = data_type
        #     id_ = example["id"]
        #     dataname_map[dataname].append(id_)
        #     if data_type == "ner":
        #         golden[id_] = example["target"].split("|")
        #     else:
        #         golden[id_] = example["target"]
        #
        # # for dataname, data_ids in dataname_map.items():
        # metric = datatype2metrics[dataname_type[dataname]]()
        # gold = {k: v for k, v in golden.items() if k in data_ids}
        # pred = {k: v for k, v in predictions.items() if k in data_ids}
        # score = metric.calc_metric(golden=gold, predictions=pred)
        # acc, f1 = score["acc"], score["f1"]
        # if len(gold) != len(pred) or len(gold) < 20:
        #     print(dataname, dataname_type[dataname], round(acc, 4), len(gold), len(pred), data_ids)
        # all_metrics["macro_f1"] += f1
        # all_metrics["micro_f1"] += f1 * len(data_ids)
        # all_metrics["eval_num"] += len(data_ids)
        # all_metrics[dataname] = round(acc, 4)
        # # all_metrics["macro_f1"] = round(all_metrics["macro_f1"] / len(dataname_map), 4)
        # # all_metrics["micro_f1"] = round(all_metrics["micro_f1"] / all_metrics["eval_num"], 4)
        return all_metrics

    def metrics_by_entity(self, label_class_span, pred_class_span):
        """
        return entity level count of total prediction, true labels, and correct prediction
        """
        # pred_class_span # {label:[(start_pos, end_pos), ...]}
        # label_class_span # {label:[(start_pos, end_pos), ...]}
        def get_cnt(label_class_span):
            """
            return the count of entities
            """
            cnt = 0
            for label in label_class_span:
                cnt += len(label_class_span[label])
            return cnt

        def get_intersect_by_entity(pred_class_span, label_class_span):
            """
            return the count of correct entity
            """
            cnt = 0
            for label in label_class_span:
                cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label, [])))))
            return cnt

        pred_cnt = get_cnt(pred_class_span)
        label_cnt = get_cnt(label_class_span)
        correct_cnt = get_intersect_by_entity(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    def save_result(self, logits, label_ids):
        self.compute_metrics((logits, ), stage="test")





"""
Processing data for FewNERD dataset based on spanproto
- spanproto is span-based prototypical network, which aims to classify each span via prototypical learning.

"""
class SpanProtoFewNERDProcessor(FewShotNERProcessor):
    def __init__(self, data_args, training_args, model_args, tokenizer=None, post_tokenizer=False, keep_raw_data=True):
        super().__init__(data_args, training_args, model_args, tokenizer, post_tokenizer=post_tokenizer, keep_raw_data=keep_raw_data)
        param = {p.split("=")[0]: p.split("=")[1] for p in (data_args.user_defined).split(" ")} # user_defined parameter
        N, Q, K, mode = param["N"], param["Q"], param["K"], param["mode"] # N: num class, Q: query entity num, K: support entity num
        self.train_file = os.path.join(data_args.data_dir, "train_{}_{}.jsonl".format(N, K))
        self.dev_file = os.path.join(data_args.data_dir, "dev_{}_{}.jsonl".format(N, K))
        self.test_file = os.path.join(data_args.data_dir, "test_{}_{}.jsonl".format(N, K))

        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.sentence1_key = None
        self.mode = mode
        self.num_class = int(N)
        self.num_example = int(K)

        self.output_dir = "./outputs/{}-{}-{}".format(self.mode, self.num_class, self.num_example)


    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForSpanProto(
            self.tokenizer,
            num_class=self.num_class,
            num_example=self.num_example,
            mode=self.mode,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            pad_to_max_length=self.data_args.pad_to_max_length,
        )

    def __load_data_from_file__(self, filepath):
        with open(filepath)as f:
            lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = json.loads(lines[i].strip())
        return lines

    def get_examples(self, set_type):
        if set_type == "train":
            examples = self._create_examples(self.__load_data_from_file__(self.train_file), set_type)
            self.train_examples = examples
        elif set_type == "dev":
            examples = self._create_examples(self.__load_data_from_file__(self.dev_file), set_type)
            self.dev_examples = examples
        elif set_type == "test":
            examples = self._create_examples(self.__load_data_from_file__(self.test_file), set_type)
            self.test_file = examples
        else:
            examples = None
        return examples

    def get_sentence_with_span(self, data, label2id):
        # 给定一个support/query set，获取每个句子对应的实体span
        word_list = data["word"]
        label_list = data["label"]
        input_texts = list()
        labeled_spans = list()
        labeled_types = list()
        for words, labels in zip(word_list, label_list): # 遍历每个句子，每个句子是以word列表存储的
            start, end = -1, -1
            current_label = ""
            text = ""
            spans = list()
            span_types = list()
            for ei, word in enumerate(words):
                label = labels[ei]
                if label == "O":  # 如果当前的word不是实体，则直接累加到text上。如果此时start不为-1，说明这是实体后的第一个O
                    text += word + " "
                    if start != -1:
                        spans.append([start, end])
                        span_types.append(label2id[current_label])
                        start, end = -1, -1
                        current_label = ""
                else:  # 如果当前的word是一个实体
                    # 如果当前的实体和上一个实体不同，则需要把上一个实体保存到span里，重新记录当前新的实体
                    if label != current_label and start != -1:
                        spans.append([start, end])
                        span_types.append(label2id[current_label])
                        start, end = -1, -1
                        current_label = ""
                    if start == -1:  # 说明当前的word是当前实体的第一个词，记住这个起始位置
                        start = len(text)
                    text += word + " "  # 将当前的word追加到text上
                    end = len(text)  # 记录一下当前的end位置
                    current_label = label
            if start != -1:
                spans.append([start, end])
                span_types.append(label2id[current_label])
                # start, end = -1, -1
                # current_label = ""
            input_texts.append(text.strip())
            labeled_spans.append(spans)
            labeled_types.append(span_types)
        return input_texts, labeled_spans, labeled_types



    def _create_examples(self, lines, set_type):
        examples = []
        # is_train = 0 if set_type == "test" else 1
        for id_, line in enumerate(lines): # 遍历每一行（每一行表示一个episode）
            target_classes = line["types"] # 当前episode对应的所有类别 list
            label2id = {v: ei for ei, v in enumerate(target_classes)}
            support = line["support"]
            query = line["query"]
            support_input_texts, support_labeled_spans, support_labeled_types = self.get_sentence_with_span(support, label2id)
            query_input_texts, query_labeled_spans, query_labeled_types = self.get_sentence_with_span(query, label2id)

            examples.append(
                {
                    "id": id_,
                    "support_input_texts": support_input_texts,
                    "support_labeled_spans": support_labeled_spans,
                    "support_labeled_types": support_labeled_types,
                    "support_sentence_num": len(support_input_texts),
                    "query_input_texts": query_input_texts,
                    "query_labeled_spans": query_labeled_spans,
                    "query_labeled_types": query_labeled_types,
                    "query_sentence_num": len(query_input_texts),
                    "stage": set_type,
                }
            )

        return examples

    def set_config(self, config):
        config.ent_type_size = 1
        config.inner_dim = 64
        config.RoPE = True
        return config

    def build_preprocess_function(self):
        # Tokenize the texts
        support_key, query_key = self.support_key, self.query_key
        # input_column = self.input_column
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length
        # label_to_id = self.label_to_id
        # label_column = self.label_column

        """
        examples = {
            "id": [...]
            "support_input_texts": [
                ["sent1", "sent2", ...], # episode1
                ["sent1", "sent2", ...], # episode2
                ...
            ]
            "support_labeled_spans": [[[x, x], xxx], ...],
            "support_labeled_types": [[..]...],
            "support_sentence_num": [...],
            "query_input_texts": ["xxx", "xxx"],
            "query_labeled_spans": [[...]...],
            "query_labeled_types": [[..]...],
            "query_sentence_num": [...],
        }

        return

        examples = {
            "id": [...]
            "support_input_texts": [
                ["sent1", "sent2", ...], # episode1
                ["sent1", "sent2", ...], # episode2
                ...
            ]
            "support_labeled_spans": [[[x, x], xxx], ...],
            "support_labeled_types": [[..]...],
            "support_sentence_num": [...],
            "query_input_texts": ["xxx", "xxx"],
            "query_labeled_spans": [[...]...],
            "query_labeled_types": [[..]...],
            "query_sentence_num": [...],
        }
        """

        def func(examples):
            features = {
                "id": examples["id"],
                "support_input": list(),
                "query_input": list(),
            }
            support_inputs, query_inputs = examples[support_key], examples[query_key]
            for ei, support_input in enumerate(support_inputs):
                # 对每个episode，对support和query进行分词
                support_input = (support_input, )
                query_input = (query_inputs[ei], )
                support_result = tokenizer(
                    *support_input,
                    padding=False,
                    max_length=max_seq_length,
                    truncation="longest_first",
                    add_special_tokens=True,
                    return_offsets_mapping=True
                )
                query_result = tokenizer(
                    *query_input,
                    padding=False,
                    max_length=max_seq_length,
                    truncation="longest_first",
                    add_special_tokens=True,
                    return_offsets_mapping=True
                )
                features["support_input"].append(support_result)
                features["query_input"].append(query_result)
            features["support_labeled_spans"] = examples["support_labeled_spans"]
            features["support_labeled_types"] = examples["support_labeled_types"]
            features["support_sentence_num"] = examples["support_sentence_num"]
            features["query_labeled_spans"] = examples["query_labeled_spans"]
            features["query_labeled_types"] = examples["query_labeled_types"]
            features["query_sentence_num"] = examples["query_sentence_num"]

            return features

        return func

    def fush_multi_answer(self, has_answer, new_answer):
        # 对于某个id测试集，出现多个example时（例如同一个测试样本使用了多个模板而生成了多个example），此时将预测的topk结果进行合并
        # has为已经合并的结果，new为当前新产生的结果，
        # has格式为 {"ans": {"prob": float(prob[index_ids[ei]]), "pos": (s, e)}, ...}
        # new {"ans": {"prob": float(prob[index_ids[ei]]), "pos": (s, e)}, ...}
        # print("has_answer=", has_answer)
        for ans, value in new_answer.items():
            if ans not in has_answer.keys():
                has_answer[ans] = value
            else:
                has_answer[ans]["prob"] += value["prob"]
                has_answer[ans]["pos"].extend(value["pos"])
        return has_answer



    def get_predict_result(self, logits, examples, stage="dev"):
        """
        logits: 表示模型除了loss部分的输出
            query_spans: list = None # e.g. [[[[1, 3], [6, 9]], ...], ...] # 表示每个episode，每个句子预测的所有span
            proto_logits: list = None # e.g. [[[0, 3], ...], ...] # 表示每个episode，每个句子，每个span预测的类别
            topk_probs: torch.FloatTensor = None
            topk_indices: torch.IntTensor = None
        examples: 表示当前的样本，
            {
                    "id": xx
                    "support_input": {
                        "input_ids": [[xxx], ...],
                        "attention_mask": [[xxx], ...],
                        "token_type_ids": [[xxx], ...],
                        "offset_mapping": xxxx
                    },
                    "query_input": {
                        "input_ids": [[xxx], ...],
                        "attention_mask": [[xxx], ...],
                        "token_type_ids": [[xxx], ...],
                        "offset_mapping": xxx
                    },
                    "support_labeled_spans": [[[x, x], ..], ..],
                    "support_labeled_types": [[xx, ..], ..],
                    "support_sentence_num": xx,
                    "query_labeled_spans": [[[x, x], ..], ..],
                    "query_labeled_types": [[xx, ..], ..],
                    "query_sentence_num": xx,
                    "stage": xx,
                }
        """
        # query_spans, proto_logits, _, __ = logits
        word_size = dist.get_world_size()
        results = dict() # 所有episode的query预测的span以及对应的类别
        for i in range(word_size):
            path = os.path.join(self.output_dir, "predict", "{}_predictions_{}.npy".format(stage, i))
            # path = "./outputs2/predict/predictions_{}.npy".format(i)
            assert os.path.exists(path), "unknown path: {}".format(path)
            if os.path.exists(path):
                res = np.load(path, allow_pickle=True)[()]
                # 合并所有device的结果
                for episode_i, value in res.items():
                    results[episode_i] = value

        predictions = dict()

        for example in examples: # 遍历每个episode
            # 当前episode ground truth
            query_labeled_spans = example["query_labeled_spans"]
            query_labeled_types = example["query_labeled_types"]
            query_offset_mapping = example["query_input"]["offset_mapping"]
            id_ = example["id"]



            new_labeled_spans = list()

            # 对于query label，将字符级别的区间转换为分词级别
            for ei in range(len(query_labeled_spans)):  # 遍历每一个句子
                labeled_span = query_labeled_spans[ei]  # list # 当前句子的所有mention span（字符级别）
                offset = query_offset_mapping[ei]  # 表示tokenizer生成的token对应原始文本中字符级别的位置区间
                new_labeled_span = list()  # 当前句子的所有mention span（token级别）
                # starts, ends = feature["start"], feature["end"]
                # print("starts=", starts)
                # print("ends=", ends)
                position_map = {}
                for i, (m, n) in enumerate(offset):  # 第i个分词对应原始文本中字符级别的区间(m, n)
                    if i != 0 and m == 0 and n == 0:
                        continue
                    for k in range(m, n + 1):
                        position_map[k] = i  # 字符级别的第k个字符属于分词i

                for span in labeled_span:  # 遍历每个span
                    start, end = span
                    end -= 1
                    # MRC 没有答案时则把label指向CLS
                    # if start == 0:
                    #     # assert end == -1
                    #     labels[ei, 0, 0, 0] = 1
                    #     new_labeled_span.append([0, 0])

                    if start in position_map and end in position_map:
                        # 指定下列元素为1，说明表示第feature_id个样本的预测区间
                        new_labeled_span.append([position_map[start], position_map[end]])
                new_labeled_spans.append(new_labeled_span)


            # 获得模型预测
            pred_spans, pred_spans_ = results[id_]["spans"], list()
            pred_types, pred_types_ = results[id_]["types"], list()
            # for spans, types in zip(pred_spans, pred_types): # 遍历每一个句子
            #     # 如果预测的label为unlabeled type，则删除（即虽然预测的是实体，但并不在当前episode规定的类里，按照规则则全部视为非实体）
            #     spans_ = list()
            #     types_ = list()
            #     for ei, type in enumerate(types):
            #         if type != self.num_class:
            #             types_.append(type)
            #             spans_.append(spans[ei])
            #     pred_spans_.append(spans_)
            #     pred_types_.append(types_)

            predictions[id_] = {
                "labeled_spans": new_labeled_spans,
                "labeled_types": query_labeled_types,
                "predicted_spans": pred_spans,
                "predicted_types": pred_types
            }

            # print(" === ")
            # print("labeled_spans=", new_labeled_spans)
            # print("query_labeled_types=", query_labeled_types)
            # print("predicted_spans=", pred_spans)
            # print("predicted_types=", pred_types)

        return predictions

    def compute_metrics(self, eval_predictions, stage="dev"):
        """
        eval_predictions: huggingface
        eval_predictions[0]: logits
        eval_predictions[1]: labels
        # print("raw_datasets=", raw_datasets["validation"])
            Dataset({
                features: ["id", "support_labeled_spans", "support_labeled_types", "support_sentence_num", "query_labeled_spans", "query_labeled_types", "query_sentence_num", "stage", "support_input", "query_input"],
                num_rows: 1000
            })
        """
        all_metrics = {
            "span_precision": 0.,
            "span_recall": 0.,
            "eval_span_f1": 0,
            "class_precision": 0.,
            "class_recall": 0.,
            "eval_class_f1": 0,
        }

        examples = self.raw_datasets["validation"] if stage == "dev" else self.raw_datasets["test"]
        # golden, dataname_map, dataname_type = {}, defaultdict(list), {}
        predictions = self.get_predict_result(eval_predictions[0], examples, stage)

        # === 下面使用Few-NERD官方提供的评测方法 ===

        pred_span_cnt = 0  # pred entity cnt
        label_span_cnt = 0  # true label entity cnt
        correct_span_cnt = 0  # correct predicted entity cnt

        pred_class_cnt = 0  # pred entity cnt
        label_class_cnt = 0  # true label entity cnt
        correct_class_cnt = 0  # correct predicted entity cnt

        # 遍历每个episode
        for episode_id, predicts in predictions.items():
            query_labeled_spans = predicts["labeled_spans"]
            query_labeled_types = predicts["labeled_types"]
            pred_span = predicts["predicted_spans"]
            pred_type = predicts["predicted_types"]
            # 遍历每个句子，为每个label生成所有的span e.g. {label:[(start_pos, end_pos), ...]}
            for label_span, label_type, pred_span, pred_type in zip(
                    query_labeled_spans, query_labeled_types, pred_span, pred_type
            ):
                # 用于评价detector检测区间的效果
                label_span_dict = {0: list()}
                pred_span_dict = {0: list()}
                # 用于评价prototype分类的效果
                label_class_dict = dict()
                pred_class_dict = dict()
                # 遍历每个span
                for span, type in zip(label_span, label_type):
                    label_span_dict[0].append((span[0], span[1]))
                    if type not in label_class_dict.keys():
                        label_class_dict[type] = list()
                    label_class_dict[type].append((span[0], span[1]))

                # 遍历每个span，判断其类别，并加入到对应类别的dict中
                for span, type in zip(pred_span, pred_type):
                    pred_span_dict[0].append((span[0], span[1]))
                    if type == self.num_class or span == [0, 0]:
                        continue
                    if type not in pred_class_dict.keys():
                        pred_class_dict[type] = list()
                    pred_class_dict[type].append((span[0], span[1]))

                tmp_pred_span_cnt, tmp_label_span_cnt, correct_span = self.metrics_by_entity(
                    label_span_dict, pred_span_dict
                )

                tmp_pred_class_cnt, tmp_label_class_cnt, correct_class = self.metrics_by_entity(
                    label_class_dict, pred_class_dict
                )
                pred_span_cnt += tmp_pred_span_cnt
                label_span_cnt += tmp_label_span_cnt
                correct_span_cnt += correct_span

                pred_class_cnt += tmp_pred_class_cnt
                label_class_cnt += tmp_label_class_cnt
                correct_class_cnt += correct_class

        span_precision = correct_span_cnt / pred_span_cnt
        span_recall = correct_span_cnt / label_span_cnt
        try:
            span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall)
        except:
            span_f1 = 0.

        class_precision = correct_class_cnt / pred_class_cnt
        class_recall = correct_class_cnt / label_class_cnt
        try:
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall)
        except:
            class_f1 = 0.
        all_metrics["span_precision"], all_metrics["span_recall"], all_metrics["eval_span_f1"] = \
            span_precision, span_recall, span_f1
        all_metrics["class_precision"], all_metrics["class_recall"], all_metrics["eval_class_f1"] = \
            class_precision, class_recall, class_f1
        print("all_metrics=", all_metrics)
        # === 下面用于计算detector的效果 ===
        # 遍历每个episode，获得预测正确的span个数，

        # for example in examples:
        #     data_type = example["data_type"]
        #     dataname = "_".join(example["id"].split("_")[:-1])
        #     if dataname not in dataname_type:
        #         dataname_type[dataname] = data_type
        #     id_ = example["id"]
        #     dataname_map[dataname].append(id_)
        #     if data_type == "ner":
        #         golden[id_] = example["target"].split("|")
        #     else:
        #         golden[id_] = example["target"]
        #
        # # for dataname, data_ids in dataname_map.items():
        # metric = datatype2metrics[dataname_type[dataname]]()
        # gold = {k: v for k, v in golden.items() if k in data_ids}
        # pred = {k: v for k, v in predictions.items() if k in data_ids}
        # score = metric.calc_metric(golden=gold, predictions=pred)
        # acc, f1 = score["acc"], score["f1"]
        # if len(gold) != len(pred) or len(gold) < 20:
        #     print(dataname, dataname_type[dataname], round(acc, 4), len(gold), len(pred), data_ids)
        # all_metrics["macro_f1"] += f1
        # all_metrics["micro_f1"] += f1 * len(data_ids)
        # all_metrics["eval_num"] += len(data_ids)
        # all_metrics[dataname] = round(acc, 4)
        # # all_metrics["macro_f1"] = round(all_metrics["macro_f1"] / len(dataname_map), 4)
        # # all_metrics["micro_f1"] = round(all_metrics["micro_f1"] / all_metrics["eval_num"], 4)
        return all_metrics

    def metrics_by_entity(self, label_class_span, pred_class_span):
        """
        return entity level count of total prediction, true labels, and correct prediction
        """
        # pred_class_span # {label:[(start_pos, end_pos), ...]}
        # label_class_span # {label:[(start_pos, end_pos), ...]}
        def get_cnt(label_class_span):
            """
            return the count of entities
            """
            cnt = 0
            for label in label_class_span:
                cnt += len(label_class_span[label])
            return cnt

        def get_intersect_by_entity(pred_class_span, label_class_span):
            """
            return the count of correct entity
            """
            cnt = 0
            for label in label_class_span:
                cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label, [])))))
            return cnt

        pred_cnt = get_cnt(pred_class_span)
        label_cnt = get_cnt(label_class_span)
        correct_cnt = get_intersect_by_entity(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    def save_result(self, logits, label_ids):
        self.compute_metrics((logits, ), stage="test")


"""
Processing data for CrossNER dataset based on spanproto
- spanproto is span-based prototypical network, which aims to classify each span via prototypical learning.

"""
class SpanProtoCrossNERProcessor(FewShotNERProcessor):
    def __init__(self, data_args, training_args, model_args, post_tokenizer=False, keep_raw_data=True):
        super().__init__(data_args, training_args, model_args, post_tokenizer=post_tokenizer,
                         keep_raw_data=keep_raw_data)
        param = {p.split("=")[0]: p.split("=")[1] for p in
                 (data_args.user_defined).split(" ")}  # user_defined parameter
        N, K, ID, mode = param["N"], param["K"], param["ID"], param["mode"]  # N: num class, Q: query entity num, K: support entity num
        # mode = "xval_ner" if K == 1 else "x_val_ner_shot_5"
        # notes: in crossner, N denotes the num class of target domain
        self.train_file = os.path.join(
            data_args.data_dir, mode,
            ("ner_train_{}.json".format(ID)) if K == "1" else ("ner-train-{}-shot-5.json".format(ID))
        )
        self.dev_file = os.path.join(
            data_args.data_dir, mode,
            ("ner_valid_{}.json".format(ID)) if K == "1" else ("ner-valid-{}-shot-5.json".format(ID)))
        self.test_file = os.path.join(
            data_args.data_dir, mode,
            ("ner_test_{}.json".format(ID)) if K == "1" else ("ner-test-{}-shot-5.json".format(ID)))

        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.sentence1_key = None
        self.mode = mode
        self.num_class = None
        self.num_example = int(K)
        self.ID = ID

        self.output_dir = "./outputs/{}-{}".format(self.mode, ID)

    def get_num_class(self, data_labels):
        if data_labels == "News":
            return 4
        if data_labels == "Wiki":
            return 11
        if data_labels == "SocialMedia":
            return 6
        if data_labels == "OntoNotes":
            return 18


    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForSpanProto(
            self.tokenizer,
            num_class=self.num_class,
            num_example=self.num_example,
            mode=self.mode,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            pad_to_max_length=self.data_args.pad_to_max_length,
            path=self.output_dir
        )

    def __load_data_from_file__(self, filepath):
        with open(filepath) as f:
            raw_data = json.load(f)
        return raw_data


    def get_tokenized_datasets(self):
        raw_datasets = DatasetDict()
        if self.training_args.do_train:
            train_examples = self.get_examples("train")
            raw_datasets["train"] = DatasetK.from_dict(
                self.list_2_json(train_examples))  # [{k1: xxx, k2: xxx}, {...}] -> {k1: [xxx, xxx], k2: [xxx, xxx]}
        if self.training_args.do_eval:
            dev_examples = self.get_examples("dev")
            raw_datasets["validation"] = DatasetK.from_dict(self.list_2_json(dev_examples))
        if self.training_args.do_predict:
            test_examples = self.get_examples("test")
            raw_datasets["test"] = DatasetK.from_dict(self.list_2_json(test_examples))

        if self.post_tokenizer:
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets
            return raw_datasets
        # datasets的bug, 对于from_dict不会创建cache,需要指定cache_file_names
        # 指定了cache_file_names在_map_single中也需要cache_files不为空才能读取cache
        for key, value in raw_datasets.items():
            value.set_cache_files(["cache_local"])
        remove_columns = [self.support_key, self.query_key]
        tokenize_func = self.build_preprocess_function()
        # 多gpu, 0计算完存cache，其他load cache
        load_from_cache_file = not self.data_args.overwrite_cache if self.training_args.local_rank in [-1, 0] else True
        base_cache_dir = os.path.join(self.model_args.cache_dir,
                                      "datasets") if self.model_args.cache_dir else os.path.join(
            os.path.expanduser("~"), ".cache/huggingface/datasets/")
        cache_dir = os.path.join(base_cache_dir, self.data_args.task_name)

        os.makedirs(cache_dir, exist_ok=True)
        with self.training_args.main_process_first(desc="dataset tokenizer map"):
            raw_datasets = raw_datasets.map(
                tokenize_func,
                batched=True,
                load_from_cache_file=load_from_cache_file,
                desc="Running tokenizer on dataset",
                cache_file_names={
                    k: f"{cache_dir}/cache_{self.data_args.task_name}_{self.data_args.data_dir.split('/')[-1]}_{self.mode}_{self.ID}_{self.num_example}_{str(k)}.arrow"
                    for k in raw_datasets},
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=remove_columns
            )
            if self.keep_raw_data:
                self.raw_datasets = raw_datasets
            print("datasets=", raw_datasets)
            return raw_datasets

    def get_examples(self, set_type):
        if set_type == "train":
            examples = self._create_examples(self.__load_data_from_file__(self.train_file), set_type)
            self.train_examples = examples
        elif set_type == "dev":
            examples = self._create_examples(self.__load_data_from_file__(self.dev_file), set_type)
            self.dev_examples = examples
        elif set_type == "test":
            examples = self._create_examples(self.__load_data_from_file__(self.test_file), set_type)
            self.test_file = examples
        else:
            examples = None
        return examples

    def get_sentence_with_span(self, data, label2id):
        # 给定一个support/query set，获取每个句子对应的实体span
        word_list = data["seq_ins"]
        label_list = data["seq_outs"]
        input_texts = list()
        labeled_spans = list()
        labeled_types = list()
        for words, labels in zip(word_list, label_list):  # 遍历每个句子，每个句子是以word列表存储的
            start, end = -1, -1
            current_label = ""
            text = ""
            spans = list()
            span_types = list()
            for ei, word in enumerate(words):
                label = labels[ei]
                label = label.replace("B-", "").replace("I-", "")
                if label == "O":  # 如果当前的word不是实体，则直接累加到text上。如果此时start不为-1，说明这是实体后的第一个O
                    text += word + " "
                    if start != -1:
                        spans.append([start, end])
                        span_types.append(label2id[current_label])
                        start, end = -1, -1
                        current_label = ""
                else:  # 如果当前的word是一个实体
                    # 如果当前的实体和上一个实体不同，则需要把上一个实体保存到span里，重新记录当前新的实体
                    if label != current_label and start != -1:
                        spans.append([start, end])
                        span_types.append(label2id[current_label])
                        start, end = -1, -1
                        current_label = ""
                    if start == -1:  # 说明当前的word是当前实体的第一个词，记住这个起始位置
                        start = len(text)
                    text += word + " "  # 将当前的word追加到text上
                    end = len(text)  # 记录一下当前的end位置
                    current_label = label
            if start != -1:
                spans.append([start, end])
                span_types.append(label2id[current_label])
                # start, end = -1, -1
                # current_label = ""
            input_texts.append(text.strip())
            labeled_spans.append(spans)
            labeled_types.append(span_types)
        return input_texts, labeled_spans, labeled_types

    def _create_examples(self, raw_data: dict, set_type):

        def get_label2id(suport_labels, query_labels):
            label2id = dict()
            for sent in suport_labels + query_labels:
                for label in sent:
                    if label == "O":
                        continue
                    label = label.replace("B-", "").replace("I-", "")
                    if label not in label2id.keys():
                        label2id[label] = len(label2id)
            return label2id
        examples = []
        # is_train = 0 if set_type == "test" else 1
        for domain_name, domain_data in raw_data.items():
            # 遍历每个domain CrossNER的训练集有两个domain数据集，验证集和测试集分别为1个domain

            for id_, line in enumerate(domain_data):  # 遍历每一个episode
                # label2id = {v: ei for ei, v in enumerate(target_classes)}
                support = line["support"]
                query = line["batch"]
                label2id = get_label2id(support["seq_outs"], query["seq_outs"])
                support_input_texts, support_labeled_spans, support_labeled_types = self.get_sentence_with_span(support,
                                                                                                                label2id)
                query_input_texts, query_labeled_spans, query_labeled_types = self.get_sentence_with_span(query, label2id)

                num_class = self.get_num_class(domain_name)
                assert num_class == len(label2id)

                examples.append(
                    {
                        "id": id_,
                        "support_input_texts": support_input_texts,
                        "support_labeled_spans": support_labeled_spans,
                        "support_labeled_types": support_labeled_types,
                        "support_sentence_num": len(support_input_texts),
                        "query_input_texts": query_input_texts,
                        "query_labeled_spans": query_labeled_spans,
                        "query_labeled_types": query_labeled_types,
                        "query_sentence_num": len(query_input_texts),
                        "num_class": num_class,
                        "stage": set_type,
                    }
                )

        return examples

    def set_config(self, config):
        config.ent_type_size = 1
        config.inner_dim = 64
        config.RoPE = True
        return config

    def build_preprocess_function(self):
        # Tokenize the texts
        support_key, query_key = self.support_key, self.query_key
        # input_column = self.input_column
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length
        # label_to_id = self.label_to_id
        # label_column = self.label_column

        """
        examples = {
            "id": [...]
            "support_input_texts": [
                ["sent1", "sent2", ...], # episode1
                ["sent1", "sent2", ...], # episode2
                ...
            ]
            "support_labeled_spans": [[[x, x], xxx], ...],
            "support_labeled_types": [[..]...],
            "support_sentence_num": [...],
            "query_input_texts": ["xxx", "xxx"],
            "query_labeled_spans": [[...]...],
            "query_labeled_types": [[..]...],
            "query_sentence_num": [...],
        }

        return

        examples = {
            "id": [...]
            "support_input_texts": [
                ["sent1", "sent2", ...], # episode1
                ["sent1", "sent2", ...], # episode2
                ...
            ]
            "support_labeled_spans": [[[x, x], xxx], ...],
            "support_labeled_types": [[..]...],
            "support_sentence_num": [...],
            "query_input_texts": ["xxx", "xxx"],
            "query_labeled_spans": [[...]...],
            "query_labeled_types": [[..]...],
            "query_sentence_num": [...],
        }
        """

        def func(examples):
            features = {
                "id": examples["id"],
                "support_input": list(),
                "query_input": list(),
            }
            support_inputs, query_inputs = examples[support_key], examples[query_key]
            for ei, support_input in enumerate(support_inputs):
                # 对每个episode，对support和query进行分词
                support_input = (support_input,)
                query_input = (query_inputs[ei],)
                support_result = tokenizer(
                    *support_input,
                    padding=False,
                    max_length=max_seq_length,
                    truncation="longest_first",
                    add_special_tokens=True,
                    return_offsets_mapping=True
                )
                query_result = tokenizer(
                    *query_input,
                    padding=False,
                    max_length=max_seq_length,
                    truncation="longest_first",
                    add_special_tokens=True,
                    return_offsets_mapping=True
                )
                features["support_input"].append(support_result)
                features["query_input"].append(query_result)
            features["support_labeled_spans"] = examples["support_labeled_spans"]
            features["support_labeled_types"] = examples["support_labeled_types"]
            features["support_sentence_num"] = examples["support_sentence_num"]
            features["query_labeled_spans"] = examples["query_labeled_spans"]
            features["query_labeled_types"] = examples["query_labeled_types"]
            features["query_sentence_num"] = examples["query_sentence_num"]

            return features

        return func

    def fush_multi_answer(self, has_answer, new_answer):
        # 对于某个id测试集，出现多个example时（例如同一个测试样本使用了多个模板而生成了多个example），此时将预测的topk结果进行合并
        # has为已经合并的结果，new为当前新产生的结果，
        # has格式为 {"ans": {"prob": float(prob[index_ids[ei]]), "pos": (s, e)}, ...}
        # new {"ans": {"prob": float(prob[index_ids[ei]]), "pos": (s, e)}, ...}
        # print("has_answer=", has_answer)
        for ans, value in new_answer.items():
            if ans not in has_answer.keys():
                has_answer[ans] = value
            else:
                has_answer[ans]["prob"] += value["prob"]
                has_answer[ans]["pos"].extend(value["pos"])
        return has_answer

    def get_predict_result(self, logits, examples, stage="dev"):
        """
        logits: 表示模型除了loss部分的输出
            query_spans: list = None # e.g. [[[[1, 3], [6, 9]], ...], ...] # 表示每个episode，每个句子预测的所有span
            proto_logits: list = None # e.g. [[[0, 3], ...], ...] # 表示每个episode，每个句子，每个span预测的类别
            topk_probs: torch.FloatTensor = None
            topk_indices: torch.IntTensor = None
        examples: 表示当前的样本，
            {
                    "id": xx
                    "support_input": {
                        "input_ids": [[xxx], ...],
                        "attention_mask": [[xxx], ...],
                        "token_type_ids": [[xxx], ...],
                        "offset_mapping": xxxx
                    },
                    "query_input": {
                        "input_ids": [[xxx], ...],
                        "attention_mask": [[xxx], ...],
                        "token_type_ids": [[xxx], ...],
                        "offset_mapping": xxx
                    },
                    "support_labeled_spans": [[[x, x], ..], ..],
                    "support_labeled_types": [[xx, ..], ..],
                    "support_sentence_num": xx,
                    "query_labeled_spans": [[[x, x], ..], ..],
                    "query_labeled_types": [[xx, ..], ..],
                    "query_sentence_num": xx,
                    "num_class": xx,
                    "stage": xx,
                }
        """
        # query_spans, proto_logits, _, __ = logits
        word_size = dist.get_world_size()
        num_class = examples[0]["num_class"]
        results = dict()  # 所有episode的query预测的span以及对应的类别
        for i in range(word_size):
            path = os.path.join(
                self.output_dir, "predict", "{}_predictions_{}.npy".format(stage, i))
            # path = "./outputs2/predict/predictions_{}.npy".format(i)
            assert os.path.exists(path), "unknown path: {}".format(path)
            if os.path.exists(path):
                res = np.load(path, allow_pickle=True)[()]
                # 合并所有device的结果
                for episode_i, value in res.items():
                    results[episode_i] = value

        predictions = dict()

        for example in examples:  # 遍历每个episode
            # 当前episode ground truth
            query_labeled_spans = example["query_labeled_spans"]
            query_labeled_types = example["query_labeled_types"]
            query_offset_mapping = example["query_input"]["offset_mapping"]
            num_class = example["num_class"]
            id_ = example["id"]

            new_labeled_spans = list()

            # 对于query label，将字符级别的区间转换为分词级别
            for ei in range(len(query_labeled_spans)):  # 遍历每一个句子
                labeled_span = query_labeled_spans[ei]  # list # 当前句子的所有mention span（字符级别）
                offset = query_offset_mapping[ei]  # 表示tokenizer生成的token对应原始文本中字符级别的位置区间
                new_labeled_span = list()  # 当前句子的所有mention span（token级别）
                # starts, ends = feature["start"], feature["end"]
                # print("starts=", starts)
                # print("ends=", ends)
                position_map = {}
                for i, (m, n) in enumerate(offset):  # 第i个分词对应原始文本中字符级别的区间(m, n)
                    if i != 0 and m == 0 and n == 0:
                        continue
                    for k in range(m, n + 1):
                        position_map[k] = i  # 字符级别的第k个字符属于分词i

                for span in labeled_span:  # 遍历每个span
                    start, end = span
                    end -= 1
                    # MRC 没有答案时则把label指向CLS
                    # if start == 0:
                    #     # assert end == -1
                    #     labels[ei, 0, 0, 0] = 1
                    #     new_labeled_span.append([0, 0])

                    if start in position_map and end in position_map:
                        # 指定下列元素为1，说明表示第feature_id个样本的预测区间
                        new_labeled_span.append([position_map[start], position_map[end]])
                new_labeled_spans.append(new_labeled_span)

            # 获得模型预测
            pred_spans, pred_spans_ = results[id_]["spans"], list()
            pred_types, pred_types_ = results[id_]["types"], list()
            # for spans, types in zip(pred_spans, pred_types): # 遍历每一个句子
            #     # 如果预测的label为unlabeled type，则删除（即虽然预测的是实体，但并不在当前episode规定的类里，按照规则则全部视为非实体）
            #     spans_ = list()
            #     types_ = list()
            #     for ei, type in enumerate(types):
            #         if type != self.num_class:
            #             types_.append(type)
            #             spans_.append(spans[ei])
            #     pred_spans_.append(spans_)
            #     pred_types_.append(types_)

            predictions[id_] = {
                "labeled_spans": new_labeled_spans,
                "labeled_types": query_labeled_types,
                "predicted_spans": pred_spans,
                "predicted_types": pred_types,
                "num_class": num_class
            }

            # print(" === ")
            # print("labeled_spans=", new_labeled_spans)
            # print("query_labeled_types=", query_labeled_types)
            # print("predicted_spans=", pred_spans)
            # print("predicted_types=", pred_types)

        return predictions

    def compute_metrics(self, eval_predictions, stage="dev"):
        """
        eval_predictions: huggingface
        eval_predictions[0]: logits
        eval_predictions[1]: labels
        # print("raw_datasets=", raw_datasets["validation"])
            Dataset({
                features: ["id", "support_labeled_spans", "support_labeled_types", "support_sentence_num", "query_labeled_spans", "query_labeled_types", "query_sentence_num", "stage", "support_input", "query_input"],
                num_rows: 1000
            })
        """
        all_metrics = {
            "span_precision": 0.,
            "span_recall": 0.,
            "eval_span_f1": 0,
            "class_precision": 0.,
            "class_recall": 0.,
            "eval_class_f1": 0,
        }

        examples = self.raw_datasets["validation"] if stage == "dev" else self.raw_datasets["test"]
        # golden, dataname_map, dataname_type = {}, defaultdict(list), {}
        predictions = self.get_predict_result(eval_predictions[0], examples, stage)

        # === 下面使用Few-NERD官方提供的评测方法 ===

        pred_span_cnt = 0  # pred entity cnt
        label_span_cnt = 0  # true label entity cnt
        correct_span_cnt = 0  # correct predicted entity cnt

        pred_class_cnt = 0  # pred entity cnt
        label_class_cnt = 0  # true label entity cnt
        correct_class_cnt = 0  # correct predicted entity cnt

        # 遍历每个episode
        for episode_id, predicts in predictions.items():
            query_labeled_spans = predicts["labeled_spans"]
            query_labeled_types = predicts["labeled_types"]
            pred_span = predicts["predicted_spans"]
            pred_type = predicts["predicted_types"]
            num_class = predicts["num_class"]
            # 遍历每个句子，为每个label生成所有的span e.g. {label:[(start_pos, end_pos), ...]}
            for label_span, label_type, pred_span, pred_type in zip(
                    query_labeled_spans, query_labeled_types, pred_span, pred_type
            ):
                # 用于评价detector检测区间的效果
                label_span_dict = {0: list()}
                pred_span_dict = {0: list()}
                # 用于评价prototype分类的效果
                label_class_dict = dict()
                pred_class_dict = dict()
                # 遍历每个span
                for span, type in zip(label_span, label_type):
                    label_span_dict[0].append((span[0], span[1]))
                    if type not in label_class_dict.keys():
                        label_class_dict[type] = list()
                    label_class_dict[type].append((span[0], span[1]))

                # 遍历每个span，判断其类别，并加入到对应类别的dict中
                for span, type in zip(pred_span, pred_type):
                    pred_span_dict[0].append((span[0], span[1]))
                    if type == num_class or span == [0, 0]:
                        continue
                    if type not in pred_class_dict.keys():
                        pred_class_dict[type] = list()
                    pred_class_dict[type].append((span[0], span[1]))

                tmp_pred_span_cnt, tmp_label_span_cnt, correct_span = self.metrics_by_entity(
                    label_span_dict, pred_span_dict
                )

                tmp_pred_class_cnt, tmp_label_class_cnt, correct_class = self.metrics_by_entity(
                    label_class_dict, pred_class_dict
                )
                pred_span_cnt += tmp_pred_span_cnt
                label_span_cnt += tmp_label_span_cnt
                correct_span_cnt += correct_span

                pred_class_cnt += tmp_pred_class_cnt
                label_class_cnt += tmp_label_class_cnt
                correct_class_cnt += correct_class

        span_precision = correct_span_cnt / pred_span_cnt
        span_recall = correct_span_cnt / label_span_cnt
        try:
            span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall)
        except:
            span_f1 = 0.

        class_precision = correct_class_cnt / pred_class_cnt
        class_recall = correct_class_cnt / label_class_cnt
        try:
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall)
        except:
            class_f1 = 0.
        all_metrics["span_precision"], all_metrics["span_recall"], all_metrics["eval_span_f1"] = \
            span_precision, span_recall, span_f1
        all_metrics["class_precision"], all_metrics["class_recall"], all_metrics["eval_class_f1"] = \
            class_precision, class_recall, class_f1
        print("{} all_metrics=".format(stage), all_metrics)
        # === 下面用于计算detector的效果 ===
        # 遍历每个episode，获得预测正确的span个数，

        # for example in examples:
        #     data_type = example["data_type"]
        #     dataname = "_".join(example["id"].split("_")[:-1])
        #     if dataname not in dataname_type:
        #         dataname_type[dataname] = data_type
        #     id_ = example["id"]
        #     dataname_map[dataname].append(id_)
        #     if data_type == "ner":
        #         golden[id_] = example["target"].split("|")
        #     else:
        #         golden[id_] = example["target"]
        #
        # # for dataname, data_ids in dataname_map.items():
        # metric = datatype2metrics[dataname_type[dataname]]()
        # gold = {k: v for k, v in golden.items() if k in data_ids}
        # pred = {k: v for k, v in predictions.items() if k in data_ids}
        # score = metric.calc_metric(golden=gold, predictions=pred)
        # acc, f1 = score["acc"], score["f1"]
        # if len(gold) != len(pred) or len(gold) < 20:
        #     print(dataname, dataname_type[dataname], round(acc, 4), len(gold), len(pred), data_ids)
        # all_metrics["macro_f1"] += f1
        # all_metrics["micro_f1"] += f1 * len(data_ids)
        # all_metrics["eval_num"] += len(data_ids)
        # all_metrics[dataname] = round(acc, 4)
        # # all_metrics["macro_f1"] = round(all_metrics["macro_f1"] / len(dataname_map), 4)
        # # all_metrics["micro_f1"] = round(all_metrics["micro_f1"] / all_metrics["eval_num"], 4)
        return all_metrics

    def metrics_by_entity(self, label_class_span, pred_class_span):
        """
        return entity level count of total prediction, true labels, and correct prediction
        """

        # pred_class_span # {label:[(start_pos, end_pos), ...]}
        # label_class_span # {label:[(start_pos, end_pos), ...]}
        def get_cnt(label_class_span):
            """
            return the count of entities
            """
            cnt = 0
            for label in label_class_span:
                cnt += len(label_class_span[label])
            return cnt

        def get_intersect_by_entity(pred_class_span, label_class_span):
            """
            return the count of correct entity
            """
            cnt = 0
            for label in label_class_span:
                cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label, [])))))
            return cnt

        pred_cnt = get_cnt(pred_class_span)
        label_cnt = get_cnt(label_class_span)
        correct_cnt = get_intersect_by_entity(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    def save_result(self, logits, label_ids):
        self.compute_metrics((logits,), stage="test")
