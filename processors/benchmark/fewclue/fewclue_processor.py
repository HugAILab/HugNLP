# -*- coding: utf-8 -*-
# @Time    : 2022/9/5 00:20 a.m.
# @Author  : JianingWang
# @File    : fewclue_processor.py


import logging
import os
import random
import pandas as pd
import torch
from processors.benchmark.fewclue.utils import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger(__name__)

""" CLUE processors and helpers """

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


class TextClassificationProcessor(DataProcessor):

    def __init__(self, task_name):
        self.task_name = task_name

    def get_train_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), header=None).values.tolist(), "train")
        return self.create_examples(self._read_json(os.path.join(data_dir, "train_few_all.json")), "train")


    def get_dev_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), header=None).values.tolist(), "dev")
        return self.create_examples(self._read_json(os.path.join(data_dir, "dev_few_all.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).values.tolist(), "test")
        return self.create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        # TODO: 文本分类数据集在此添加 labels，下同 (这里指原数据集中出现的 label)
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
        elif self.task_name == "eprstmt":
            return ["Negative", "Positive"]
        elif self.task_name == "iflytek":
            return list(range(119))
        elif self.task_name == "tnews":
            return [100, 101, 102, 103, 104, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116]
        elif self.task_name == "csldcp":
            return ["材料科学与工程", "作物学", "口腔医学", "药学", "教育学", "水利工程", "理论经济学", "食品科学与工程", "畜牧学/兽医学", "体育学", "核科学与技术", "力学", "园艺学", "水产", "法学", "地质学/地质资源与地质工程", "石油与天然气工程", "农林经济管理", "信息与通信工程", "图书馆、情报与档案管理", "政治学", "电气工程", "海洋科学", "民族学", "航空宇航科学与技术", "化学/化学工程与技术", "哲学", "公共卫生与预防医学", "艺术学", "农业工程", "船舶与海洋工程", "计算机科学与技术", "冶金工程", "交通运输工程", "动力工程及工程热物理", "纺织科学与工程", "建筑学", "环境科学与工程", "公共管理", "数学", "物理学", "林学/林业工程", "心理学", "历史学", "工商管理", "应用经济学", "中医学/中药学", "天文学", "机械工程", "土木工程", "光学工程", "地理学", "农业资源利用", "生物学/生物科学与工程", "兵器科学与技术", "矿业工程", "大气科学", "基础医学/临床医学", "电子科学与技术", "测绘科学与技术", "控制科学与工程", "军事学", "中国语言文学", "新闻传播学", "社会学", "地球物理学", "植物保护"]
        elif self.task_name == "cluewsc":
            return [False, True]
        else:
            raise Exception("task_name not supported.")

    def get_verbalizers(self):
        if self.task_name == "eprstmt":
            return {"0": "消极", "1": "积极"}
        elif self.task_name == "csldcp":
            label_list = self.get_labels()
            return {str(ei): k for ei, k in enumerate(label_list)}
        return None

    def create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        # TODO: example 添加语句
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if self.task_name in ["eprstmt"]:
                # examples.append(InputExample(guid=guid, text_a=line["sentence"], label=line["label"]))
                examples.append({
                    "guid": guid,
                    "text_a": line["sentence"],
                    "text_b": None,
                    "label": label_map[line["label"]] if set_type != "test" else "",
                    "is_train": 1 if set_type != "test" else 0
                })
            elif self.task_name in ["csldcp"]:
                # examples.append(InputExample(guid=guid, text_a=line["context"], label=line["label"]))
                examples.append({
                    "guid": guid,
                    "text_a": line["context"],
                    "text_b": None,
                    "label": label_map[line["label"]] if set_type != "test" else "",
                    "is_train": 1 if set_type != "test" else 0
                })
            else:
                raise Exception("Task_name not supported.")

        return examples


class TnewsProcessor(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train_few_all.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev_few_all.json")), "dev")

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

    def get_verbalizers(self):
        v = {
            0: ["故事"], 1: ["文化"], 2: ["娱乐"], 3: ["体育"], 4: ["财经"], 5: ["房产"], 6: ["汽车"],
            7: ["教育"], 8: ["科技"], 9: ["军事"], 10: ["旅游"], 11: ["国际"], 12: ["股票"], 13: ["农业"],
            14: ["电竞"]
        }
        v = {str(key): value[0] for key, value in v.items()}
        return v

    def create_examples(self, lines, set_type, use_keys=False):
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["sentence"]
            text_b = None
            label = str(line["label"]) if set_type != "test" else "100"
            if use_keys:
                keywords = line["keywords"].replace(",", " ")
                text_a = "{} {}".format(keywords, line["sentence"])
            examples.append({
                "guid": guid,
                "text_a": text_a,
                "text_b": text_b,
                "label": label_map[label],
                "is_train": 1 if set_type != "test" else 0
            })
        return examples

class IflytekProcessor(DataProcessor):
    """Processor for the IFLYTEK data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train_few_all.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev_few_all.json")), "dev")

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

    def get_verbalizers(self):
        v = {
            0: ["打车"], 100: ["美颜"], 101: ["影像"], 102: ["摄影"], 103: ["相机"], 104: ["绘画"], 105: ["二手"], 106: ["电商"],
            107: ["团购"], 108: ["外卖"], 109: ["票务"], 10: ["社区"], 110: ["超市"], 111: ["购物"], 112: ["笔记"], 113: ["办公"],
            114: ["日程"], 115: ["女性"], 116: ["经营"], 117: ["收款"], 118: ["其他"], 11: ["赚钱"], 12: ["魔幻"], 13: ["仙侠"],
            14: ["卡牌"], 15: ["飞行"], 16: ["射击"], 17: ["休闲"], 18: ["动作"], 19: ["体育"], 1: ["地图"], 20: ["棋牌"],
            21: ["养成"], 22: ["策略"], 23: ["竞技"], 24: ["辅助"], 25: ["约会"], 26: ["通讯"], 27: ["工作"], 28: ["论坛"],
            29: ["婚恋"], 2: ["免费"], 30: ["情侣"], 31: ["社交"], 32: ["生活"], 33: ["博客"], 34: ["新闻"], 35: ["漫画"],
            36: ["小说"], 37: ["技术"], 38: ["教辅"], 39: ["问答"], 3: ["租车"], 40: ["搞笑"], 41: ["杂志"], 42: ["百科"],
            43: ["影视"], 44: ["求职"], 45: ["兼职"], 46: ["视频"], 47: ["短视"], 48: ["音乐"], 49: ["直播"], 4: ["同城"],
            50: ["电台"], 51: ["唱歌"], 52: ["两性"], 53: ["小学"], 54: ["职考"], 55: ["公务"], 56: ["英语"], 57: ["在线"],
            58: ["教育"], 59: ["成人"], 5: ["快递"], 60: ["艺术"], 61: ["语言"], 62: ["旅游"], 63: ["预定"], 64: ["民航"],
            65: ["铁路"], 66: ["酒店"], 67: ["行程"], 68: ["民宿"], 69: ["出国"], 6: ["婚庆"], 70: ["工具"], 71: ["亲子"],
            72: ["母婴"], 73: ["驾校"], 74: ["违章"], 75: ["汽车"], 76: ["买车"], 77: ["养车"], 78: ["行车"], 79: ["租房"],
            7: ["家政"], 80: ["买房"], 81: ["装修"], 82: ["电子"], 83: ["挂号"], 84: ["养生"], 85: ["医疗"], 86: ["减肥"],
            87: ["美妆"], 88: ["菜谱"], 89: ["餐饮"], 8: ["交通"], 90: ["资讯"], 91: ["运动"], 92: ["支付"], 93: ["保险"],
            94: ["股票"], 95: ["借贷"], 96: ["理财"], 97: ["彩票"], 98: ["记账"], 99: ["银行"], 9: ["政务"]
        }
        v = {str(key): value[0] for key, value in v.items()}
        return v

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["sentence"]
            text_b = None
            label = str(line["label"]) if set_type != "test" else "0"
            examples.append({
                "guid": guid,
                "text_a": text_a,
                "text_b": text_b,
                "label": label_map[label],
                "is_train": 1 if set_type != "test" else 0
            })
        return examples

class OcnliProcessor(DataProcessor):
    """Processor for the CMNLI data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_lines = self._read_json(os.path.join(data_dir, "train_few_all.json"))
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
            self._read_json(os.path.join(data_dir, "dev_few_all.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def get_verbalizers(self):
        return {
            "contradiction": "矛盾",
            "entailment": "蕴含",
            "neutral": "中立",
        }

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
                label = str(line["label"]) if set_type != "test" else "neutral"
                if label.strip()=="-":
                    continue
                examples.append({
                    "guid": guid,
                    "text_a": text_a,
                    "text_b": text_b,
                    "label": label_map[label],
                    "pseudo_proba": 1.0,
                    "is_train": 1 if set_type != "test" else 0
                })
                # examples.append({
                #     "guid": guid,
                #     "text_a": text_b,
                #     "text_b": text_a,
                #     "label": label_map[label],
                #     "pseudo_proba": 1.0,
                #     "is_train": 1 if set_type != "test" else 0
                # })
        return examples


class CslProcessor(DataProcessor):
    """Processor for the CSL data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train_few_all.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev_few_all.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_verbalizers(self):
        return {
            "0": "可以",
            "1": "不可以",
        }

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = " ".join(line["keyword"])
            text_b = line["abst"]
            label = str(line["label"]) if set_type != "test" else "0"
            examples.append({
                "guid": guid,
                "text_a": text_a,
                "text_b": text_b,
                "label": label_map[label],
                "is_train": 1 if set_type != "test" else 0
            })
        return examples



class WscProcessor(DataProcessor):
    """Processor for the WSC data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "train_few_all.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "dev_few_all.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["true", "false"]

    def get_verbalizers(self):
        return {
            "true": "是的",
            "false": "不是",
        }

    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["text"]
            text_a_list = list(text_a)
            target = line["target"]
            query = target["span1_text"]
            query_idx = target["span1_index"]
            pronoun = target["span2_text"]
            pronoun_idx = target["span2_index"]
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
            label = str(line["label"]) if set_type != "test" else "true"
            # examples.append({
            #     "guid": guid,
            #     "text_a": text_a,
            #     "text_b": text_b,
            #     "label": label_map[label],
            #     "is_train": 1 if set_type != "test" else 0
            # })
            # print([s1, e1, s2, e2])
            examples.append({
                "guid": guid,
                "text_a": text_a,
                "text_b": text_b,
                # "span": [s1+1, e1+1, s2+1, e2+1],
                "label": label_map[label],
                "is_train": 1 if set_type != "test" else 0
            })
        return examples


class BustmProcessor(DataProcessor):
    """Processor for the BUSTM data set."""

    # def get_example_from_tensor_dict(self, tensor_dict):
        # """See base class."""
        # return InputExample(
            # tensor_dict["idx"].numpy(),
            # tensor_dict["question"].numpy().decode("utf-8"),
            # tensor_dict["sentence"].numpy().decode("utf-8"),
            # str(tensor_dict["label"].numpy()),
        # )

    def get_train_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(pd.read_csv(os.path.join(data_dir, "train.csv"), header=None).values.tolist(), "train")
        return self.create_examples(self._read_json(os.path.join(data_dir, "train_few_all.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(pd.read_csv(os.path.join(data_dir, "dev.csv"), header=None).values.tolist(), "dev")
        return self.create_examples(self._read_json(os.path.join(data_dir, "dev_few_all.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).values.tolist(), "test")
        return self.create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return list(range(2))

    def get_verbalizers(self):
        return {
            "0": "不相似",
            "1": "相似",
        }

    def create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        label_list = self.get_labels()
        label_map = {str(label): i for i, label in enumerate(label_list)}
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["sentence1"]
            text_b = line["sentence2"]

            label = line["label"] if "label" in line.keys() else ""
            # examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            examples.append({
                "guid": guid,
                "text_a": text_a,
                "text_b": text_b,
                "label": label_map[label] if "label" in line.keys() else "",
                "is_train": 1 if set_type != "test" else 0
            })
        return examples


clue_tasks_num_labels = {
    "iflytek": 119,
    "ocnli": 3,
    "csl": 2,
    "wsc": 2,
    "tnews": 15,
    "eprstmt": 2,
    "csldcp": 67,
    "bustm": 2,
    "chid": None,
}

clue_processors = {
    "tnews": TnewsProcessor(),
    "iflytek": IflytekProcessor(),
    "ocnli": OcnliProcessor(),
    "csl": CslProcessor(),
    "wsc": WscProcessor(),
    "eprstmt": TextClassificationProcessor("eprstmt"),
    "csldcp": TextClassificationProcessor("csldcp"),
    "bustm": BustmProcessor(),
    "chid": None,
}

clue_output_modes = {
    "tnews": "classification",
    "iflytek": "classification",
    "ocnli": "classification",
    "csl": "classification",
    "wsc": "classification",
    "eprstmt": "classification",
    "csldcp": "classification",
    "bustm": "classification",
    "chid": None,
}

clue_task_to_instruction_type = {
    "tnews": "cls",
    "iflytek": "cls",
    "ocnli": "nli",
    "csl": "cslkeys",
    "wsc": "wsc",
    "eprstmt": "cls",
    "csldcp": "cls",
    "bustm": "similarity",
    "chid": "chid",
}
