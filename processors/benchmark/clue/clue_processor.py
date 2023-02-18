# -*- coding: utf-8 -*-
# @Author: bo.shi
# @Date:   2019-12-30 19:26:53
# @Last Modified by:   bo.shi
# @Last Modified time: 2020-01-01 11:39:23
""" CLUE processors and helpers """

import logging
import os
import random

import torch
from processors.benchmark.clue.utils import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger(__name__)


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def xlnet_collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, -max_len:]
    all_attention_mask = all_attention_mask[:, -max_len:]
    all_token_type_ids = all_token_type_ids[:, -max_len:]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def clue_convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: CLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if task is not None:
        processor = clue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = clue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)
        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))
            logger.info("input length: %d" % (input_len))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label,
                          input_len=input_len))
    return features


class TnewsProcessor(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        labels = []
        for i in range(17):
            if i == 5 or i == 11:
                continue
            labels.append(str(100 + i))
        return labels

    def create_examples(self, lines, set_type, use_keys=False):
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence']
            text_b = None
            label = str(line['label']) if set_type != 'test' else "100"
            if use_keys:
                keywords = line['keywords'].replace(',', ' ')
                text_a = "{} {}".format(keywords, line['sentence'])
            examples.append({
                "guid": guid,
                "text_a": text_a,
                "text_b": text_b,
                "label": label_map[label],
                "is_train": 1 if set_type != 'test' else 0
            })
        return examples



class TnewsEFLProcessor(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        labels = []
        for i in range(17):
            if i == 5 or i == 11:
                continue
            labels.append(str(100 + i))
        return labels

    def create_examples(self, lines, set_type, use_keys=True, label_desc: dict=None):
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence']
            text_b = None
            label = str(line['label']) if set_type != 'test' else "100"
            keywords = line['keywords'].replace(',', '，')
            # 一共有15个class，因此每个样本可以构建15个example
            for label_class, desc_text in label_desc.items():

                if use_keys:
                    text_a = "{}。{}。{}".format(
                        "这是一篇关于【{}】的新闻".format(desc_text),
                        keywords, line['sentence']
                    )
                else:
                    text_a = "{}。{}".format(
                        "这是一篇关于【{}】的新闻".format(desc_text),
                        line['sentence']
                    )
                examples.append({
                    "guid": guid,
                    "text_a": text_a,
                    "text_b": text_b,
                    "origin_label": label_map[label],
                    "label": 1 if label == label_class else 0,
                    "is_train": 1 if set_type != 'test' else 0
                })
        return examples


class IflytekProcessor(DataProcessor):
    """Processor for the IFLYTEK data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        labels = []
        for i in range(119):
            labels.append(str(i))
        return labels

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence']
            text_b = None
            label = str(line['label']) if set_type != 'test' else "0"
            examples.append({
                "guid": guid,
                "text_a": text_a,
                "text_b": text_b,
                "label": label_map[label],
                "is_train": 1 if set_type != 'test' else 0
            })
        return examples


class AfqmcProcessor(DataProcessor):
    """Processor for the AFQMC data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_lines = self._read_json(os.path.join(data_dir, "train.json"))
        if os.path.exists(os.path.join(data_dir, "dev_pseudo.json")):
            dev_pseudo_lines = self._read_json(os.path.join(data_dir, "dev_pseudo.json"))
            train_lines.extend(dev_pseudo_lines)
        if os.path.exists(os.path.join(data_dir, "test_pseudo.json")):
            test_pseudo_lines = self._read_json(os.path.join(data_dir, "test_pseudo.json"))
            train_lines.extend(test_pseudo_lines)
        return self.create_examples(train_lines, "train")


    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def create_examples(self, lines, set_type):
        # high_frequency = ["花呗", "借呗"]
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if "pseudo_proba" in line.keys():
                examples.append({
                    "guid": line["guid"],
                    "text_a": line["text_a"],
                    "text_b": line["text_b"],
                    "label": int(line["label"]),
                    "pseudo_proba": float(line["pseudo_proba"]),
                    "is_train": 1
                })
                examples.append({
                    "guid": line["guid"],
                    "text_a": line["text_b"],
                    "text_b": line["text_a"],
                    "label": int(line["label"]),
                    "pseudo_proba": float(line["pseudo_proba"]),
                    "is_train": 1
                })
            else:
                text_a = line['sentence1']
                text_b = line['sentence2']
                # 去掉一些高频词以避免对模型产生影响
                # text_a = text_a.replace(high_frequency[0], "").replace(high_frequency[1], "")
                # text_b = text_b.replace(high_frequency[0], "").replace(high_frequency[1], "")
                label = str(line['label']) if set_type != 'test' else "0"
                pseudo_proba = 1.0

                # examples.append(
                #     InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                examples.append({
                    "guid": guid,
                    "text_a": text_a,
                    "text_b": text_b,
                    "label": label_map[label],
                    "pseudo_proba": pseudo_proba,
                    "is_train": 1 if set_type != 'test' else 0
                })
                examples.append({
                    "guid": guid,
                    "text_a": text_b,
                    "text_b": text_a,
                    "label": label_map[label],
                    "pseudo_proba": pseudo_proba,
                    "is_train": 1 if set_type != 'test' else 0
                })
        if set_type != "test":
            random.shuffle(examples)
        return examples



class TextSimilarityProcessor(DataProcessor):
    """Processor for the AFQMC data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence1']
            text_b = line['sentence2']
            label = str(line['label']) if set_type != 'test' else "0"
            # examples.append(
            #     InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            examples.append({
                "guid": guid,
                "text_a": text_a,
                "text_b": text_b,
                "label": label_map[label],
                "is_train": 1 if set_type != 'test' else 0
            })
        return examples

class OcnliProcessor(DataProcessor):
    """Processor for the CMNLI data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_lines = self._read_json(os.path.join(data_dir, "train.json"))
        if os.path.exists(os.path.join(data_dir, "dev_pseudo.json")):
            dev_pseudo_lines = self._read_json(os.path.join(data_dir, "dev_pseudo.json"))
            train_lines.extend(dev_pseudo_lines)
        if os.path.exists(os.path.join(data_dir, "test_pseudo.json")):
            test_pseudo_lines = self._read_json(os.path.join(data_dir, "test_pseudo.json"))
            train_lines.extend(test_pseudo_lines)
        return self.create_examples(train_lines, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if "pseudo_proba" in line.keys():
                examples.append({
                    "guid": line["guid"],
                    "text_a": line["text_a"],
                    "text_b": line["text_b"],
                    "label": int(line["label"]),
                    "pseudo_proba": float(line["pseudo_proba"]),
                    "is_train": 1
                })
                # examples.append({
                #     "guid": line["guid"],
                #     "text_a": line["text_b"],
                #     "text_b": line["text_a"],
                #     "label": int(line["label"]),
                #     "pseudo_proba": float(line["pseudo_proba"]),
                #     "is_train": 1
                # })
            else:
                text_a = line["sentence1"]
                text_b = line["sentence2"]
                label = str(line["label"]) if set_type != 'test' else 'neutral'
                if label.strip()=='-':
                    continue
                examples.append({
                    "guid": guid,
                    "text_a": text_a,
                    "text_b": text_b,
                    "label": label_map[label],
                    "pseudo_proba": 1.0,
                    "is_train": 1 if set_type != 'test' else 0
                })
                # examples.append({
                #     "guid": guid,
                #     "text_a": text_b,
                #     "text_b": text_a,
                #     "label": label_map[label],
                #     "pseudo_proba": 1.0,
                #     "is_train": 1 if set_type != 'test' else 0
                # })
        return examples


class CmnliProcessor(DataProcessor):
    """Processor for the CMNLI data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["sentence1"]
            text_b = line["sentence2"]
            label = str(line["label"]) if set_type != 'test' else 'neutral'
            if label.strip()=='-':
                continue
            examples.append({
                "guid": guid,
                "text_a": text_a,
                "text_b": text_b,
                "label": label_map[label],
                "is_train": 1 if set_type != 'test' else 0
            })
        return examples


class CslProcessor(DataProcessor):
    """Processor for the CSL data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = " ".join(line['keyword'])
            text_b = line['abst']
            label = str(line['label']) if set_type != 'test' else '0'
            examples.append({
                "guid": guid,
                "text_a": text_a,
                "text_b": text_b,
                "label": label_map[label],
                "is_train": 1 if set_type != 'test' else 0
            })
        return examples



class CslEFLProcessor(DataProcessor):
    """Processor for the CSL data set (CLUE version)."""
    # 不同于传统方法，对相同的文本，罗列出所有标注的关键词，每个关键词与文本组成一个example

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []


        if set_type == 'train':
            abst2keys = dict()  # 每个文本对应的所有关键词及其标签
            # 先将标签为1的全部保存下来，因为1对应的所有关键词一定是正确的
            for (i, line) in enumerate(lines):
                # guid = "%s-%s" % (set_type, i)
                keywords = line['keyword'] # list
                text_b = line['abst']
                # label = str(line['label']) if set_type != 'test' else '0'

                label = str(line['label'])
                if label == "1":
                    if text_b not in abst2keys.keys():
                        abst2keys[text_b] = dict()
                    # 说明当前所有的关键词都是正确的
                    for k in keywords:
                        abst2keys[text_b][k] = "1"
            # 重新扫描一遍，对那些标签为"0"的进行处理
            for (i, line) in enumerate(lines):
                # guid = "%s-%s" % (set_type, i)
                keywords = line['keyword'] # list
                text_b = line['abst']
                # label = str(line['label']) if set_type != 'test' else '0'
                if text_b not in abst2keys.keys(): # 如果不存在，则抛弃该文本，因为我们无法知道标签为0的哪些关键词是对的
                    continue
                correct_keys = abst2keys[text_b]
                label = str(line['label'])
                if label == "0":
                    for k in keywords:
                        if k not in correct_keys.keys(): # 只有标签为0，且当前的关键词不在已知的正确标签集合里，这些标签是不能表示当前文章的
                            abst2keys[text_b][k] = "0"
            # 生成新的数据集
            ei = 0
            for abst, key2label in abst2keys.items():
                guid = "%s-%s" % (set_type, ei)

                for key, label in key2label.items():
                    examples.append({
                        "guid": guid,
                        "text_a": "【{}】可以作为文章的关键词吗？文章：【{}】".format(key, abst),
                        "text_b": None,
                        "label": label_map[label],
                        "is_train": 1 if set_type != 'test' else 0,

                    })
                ei += 1
        else:
            # 对于验证集、测试集，每个example，将每个abst与关键词对应一个样本即可
            for (i, line) in enumerate(lines):
                guid = "%s-%s" % (set_type, i)
                keywords = line['keyword'] # list
                abst = line['abst']
                label = str(line['label']) if set_type != 'test' else '0'
                for key in keywords:
                    examples.append({
                        "guid": guid,
                        "text_a": "【{}】可以作为文章的关键词吗？文章：【{}】".format(key, abst),
                        "text_b": None,
                        "label": label_map[label],
                        "is_train": 0
                    })

        return examples




class WscProcessor(DataProcessor):
    """Processor for the WSC data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["true", "false"]

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['text']
            text_a_list = list(text_a)
            target = line['target']
            query = target['span1_text']
            query_idx = target['span1_index']
            pronoun = target['span2_text']
            pronoun_idx = target['span2_index']
            assert text_a[pronoun_idx: (pronoun_idx + len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
            assert text_a[query_idx: (query_idx + len(query))] == query, "query: {}".format(query)
            s1, e1, s2, e2 = 0, 0, 0, 0
            if pronoun_idx > query_idx:
                text_a_list.insert(query_idx, "_")
                text_a_list.insert(query_idx + len(query) + 1, "_")
                text_a_list.insert(pronoun_idx + 2, "[")
                text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
                s1, e1, s2, e2 = query_idx, query_idx + len(query) + 1, pronoun_idx + 2, pronoun_idx + len(pronoun) + 2 + 1
            else:
                text_a_list.insert(pronoun_idx, "[")
                text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
                text_a_list.insert(query_idx + 2, "_")
                text_a_list.insert(query_idx + len(query) + 2 + 1, "_")
                s1, e1, s2, e2 = pronoun_idx, pronoun_idx + len(pronoun) + 1, query_idx + 2, query_idx + len(query) + 2 + 1
            text_a = "".join(text_a_list) # xxx_xxx_xxxx[xx]xxxx
            text_b = None
            label = str(line['label']) if set_type != 'test' else 'true'
            # examples.append({
            #     "guid": guid,
            #     "text_a": text_a,
            #     "text_b": text_b,
            #     "label": label_map[label],
            #     "is_train": 1 if set_type != 'test' else 0
            # })
            # print([s1, e1, s2, e2])
            examples.append({
                "guid": guid,
                "text_a": text_a,
                "text_b": text_b,
                # "span": [s1+1, e1+1, s2+1, e2+1],
                "label": label_map[label],
                "is_train": 1 if set_type != 'test' else 0
            })
        return examples


class CopaProcessor(DataProcessor):
    """Processor for the COPA data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def create_examples(self, lines, set_type):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            i = 2 * i
            guid1 = "%s-%s" % (set_type, i)
            guid2 = "%s-%s" % (set_type, i + 1)
            premise = line['premise']
            choice0 = line['choice0']
            label = str(1 if line['label'] == 0 else 0) if set_type != 'test' else '0'
            choice1 = line['choice1']
            label2 = str(0 if line['label'] == 0 else 1) if set_type != 'test' else '0'
            if line['question'] == 'effect':
                text_a = premise
                text_b = choice0
                text_a2 = premise
                text_b2 = choice1
            elif line['question'] == 'cause':
                text_a = choice0
                text_b = premise
                text_a2 = choice1
                text_b2 = premise
            else:
                raise ValueError(f'unknowed {line["question"]} type')
            examples.append({
                "guid": guid1,
                "text_a": text_a,
                "text_b": text_b,
                "label": label_map[label],
                "is_train": 1 if set_type != 'test' else 0
            })
            examples.append({
                "guid": guid2,
                "text_a": text_a2,
                "text_b": text_b2,
                "label": label_map[label2],
                "is_train": 1 if set_type != 'test' else 0
            })
        return examples

    def create_examples_version2(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if line['question'] == 'cause':
                text_a = line['premise'] + '这是什么原因造成的？' + line['choice0']
                text_b = line['premise'] + '这是什么原因造成的？' + line['choice1']
            else:
                text_a = line['premise'] + '这造成了什么影响？' + line['choice0']
                text_b = line['premise'] + '这造成了什么影响？' + line['choice1']
            label = str(1 if line['label'] == 0 else 0) if set_type != 'test' else '0'
            examples.append({
                "guid": guid,
                "text_a": text_a,
                "text_b": text_b,
                "label": label,
                "is_train": 1 if set_type != 'test' else 0
            })
        return examples


class QbqtcProcessor(DataProcessor):
    """Processor for the CMNLI data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["query"]
            text_b = line["title"]
            label = str(line["label"]) if set_type != 'test' else '0'
            if label.strip()=='-':
                continue
            examples.append({
                "guid": guid,
                "text_a": text_a,
                "text_b": text_b,
                "label": label_map[label],
                "is_train": 1 if set_type != 'test' else 0
            })
        return examples



clue_tasks_num_labels = {
    'iflytek': 119,
    'cmnli': 3,
    'ocnli': 3,
    'afqmc': 2,
    'csl': 2,
    'wsc': 2,
    'copa': 2,
    'tnews': 15,
    'qbqtc': 3,
    'text_similarity': 2,
}

clue_processors = {
    'tnews': TnewsProcessor,
    'iflytek': IflytekProcessor,
    'cmnli': CmnliProcessor,
    'ocnli': OcnliProcessor,
    'afqmc': AfqmcProcessor,
    'csl': CslProcessor,
    'wsc': WscProcessor,
    'copa': CopaProcessor,
    'qbqtc': QbqtcProcessor,
    'csl_efl': CslEFLProcessor,
    'tnews_efl': TnewsEFLProcessor,
    'text_similarity': TextSimilarityProcessor,
}

clue_output_modes = {
    'tnews': "classification",
    'tnews_efl': "classification",
    'iflytek': "classification",
    'cmnli': "classification",
    'ocnli': "classification",
    'afqmc': "classification",
    'csl': "classification",
    'csl_efl': "classification",
    'wsc': "classification",
    'copa': "classification",
    'qbqtc': "classification",
    'text_similarity': 'classification',
}
