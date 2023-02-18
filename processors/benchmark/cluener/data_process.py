# -*- coding: utf-8 -*-
# @Time    : 2022/6/28 23:23 下午
# @Author  : JianingWang
# @File    : clue_ner
import random
import re
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
from processors.benchmark.cluemrc.clue_processor import clue_processors


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if '\u4e00' <= cp <= '\u9fff':  #
        return True

    if cp in "！？｡。＂＃＄％＆＇＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》（）「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏.":
        return True
    return False

def _is_eng_or_pun(cp):
    if 'a' <= cp <= 'z' or 'A' <= cp <= 'Z' \
        or cp in "！？｡。＂＃＄％＆＇＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃（「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟" \
                 "〰〾〿–—‘'‛“”„‟…‧﹏.":
        return True
    return False

def label_smooth(label_matrix, start, end, feature_id, max_len, smooth_epsilon):
    # ACL2022: Boundary Smoothing for Named Entity Recognition
    d1 = [-1, 0, 1, 0, -2, -1, 0, 1, 2, 1, 0, -1]
    d2 = [0, 1, 0, -1, 0, 1, 2, 1, 0, -1, -2, -1]
    for t1, t2 in zip(d1, d2):
        dis = abs(t1) + abs(t2)
        if start + t1 >= end + t2 and start + t1 >= 0 and end + t2 < max_len:
            if label_matrix[feature_id, 0, start + t1, end + t2] == 0:
                label_matrix[feature_id, 0, start + t1, end + t2] = smooth_epsilon / dis
            if label_matrix[feature_id, 0, start + t1, end + t2] == smooth_epsilon / 2 and dis == 1:
                label_matrix[feature_id, 0, start + t1, end + t2] = smooth_epsilon
    return label_matrix

@dataclass
class DataCollatorForGlobalPointer:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 196
    pad_to_multiple_of: Optional[int] = None
    pad_to_max_length: Optional[bool] = None
    label_smooth: Optional[bool] = False
    smooth_epsilon: Optional[float] = 0.1

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
                # print("offset=", offset)
                for i, (m, n) in enumerate(offset):
                    if i != 0 and m == 0 and n == 0:
                        continue
                    for k in range(m, n + 1):
                        position_map[k] = i # 字符级别的第k个字符属于分词i
                for start, end in zip(starts, ends):
                    start += 1
                    end += 1
                    # MRC 没有答案时则把label指向CLS
                    # print("start={}, end={}".format(start, end))
                    if start == 0:
                        assert end == 0
                        # end = -1
                        labels[feature_id, 0, 0, 0] = 1
                    else:
                        if start in position_map and end in position_map:
                            # 指定下列元素为1，说明表示第feature_id个样本的预测区间
                            s, e = position_map[start], position_map[end]
                            labels[feature_id, 0, s, e] = 1
                            if self.label_smooth:
                                labels[feature_id, 0, s, e] = 1 - self.smooth_epsilon
                                labels = label_smooth(labels, s, e, feature_id, self.max_length, self.smooth_epsilon)


            # short_labels没用，解决transformers trainer默认会merge labels导致内存爆炸的问题
            # 需配合--label_names=short_labels使用
            batch['labels'] = labels
            if batch['labels'].max() > 0:
                batch['short_labels'] = torch.ones(len(features))
            else:
                batch['short_labels'] = torch.zeros(len(features))
            return batch


class CLUENERProcessor(CLSProcessor):
    def __init__(self, data_args, training_args, model_args, post_tokenizer=False, keep_raw_data=True):
        super().__init__(data_args, training_args, model_args, post_tokenizer=post_tokenizer, keep_raw_data=keep_raw_data)
        param = {p.split("=")[0]: p.split("=")[1] for p in (data_args.user_defined).split(" ")}
        assert "data_name" in param, "You must add one defined param 'data_name=xxx' in the user_defined parameter."
        self.data_name = param["data_name"]
        self.train_file = os.path.join(data_args.data_dir, 'train.json')
        if not os.path.exists(self.train_file): # 预训练的语料
            self.train_file = os.path.join(data_args.data_dir, 'all_train.json')
        self.dev_file = os.path.join(data_args.data_dir, 'dev.json')
        self.test_file = os.path.join(data_args.data_dir, 'test.json')
        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.sentence1_key = None
        '''
        数据分为10个标签类别，分别为: 
        地址（address），书名（book），公司（company），游戏（game），政府（government），电影（movie），姓名（name），
        组织机构（organization），职位（position），景点（scene）
        '''
        self.verbalizer = {
            "address": "地名",
            "book": "书籍",
            "company": "公司",
            "game": "游戏",
            "government": "政府",
            "movie": "电影",
            "name": "人名",
            "organization": "机构",
            "position": "职位",
            "scene": "景点",
        }
        self.label_desc = {
            "address": "地名是指国家、城市、街道等地区或地理位置",
            "book": "书籍是指琴棋书画的名称",
            "company": "公司是指企业、单位、",
            "game": "游戏是指电子游戏、网络游戏",
            "government": "政府是指国家单位和权力机关",
            "movie": "电影是指供人休闲观看的视频",
            "name": "人名是指人们起的名字",
            "organization": "机构是指由多个人组成的团队或组织",
            "position": "职位是指人担任的职务名称",
            "scene": "景点是指公园、旅游区或观景点",
        }
        self.prompt = [
            "类型解释：【{}】问题：找出文章中关于【{}】类型的所有实体？文章：【{}】",
            "类型解释：【{}】问题：文章中包含【{}】类型的实体有哪些？文章：【{}】"
        ]
        self.prompt_id = 0

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForGlobalPointer(self.tokenizer, max_length=self.max_len, pad_to_multiple_of=8 if pad_to_multiple_of_8 else None, pad_to_max_length=self.data_args.pad_to_max_length)

    def get_examples(self, set_type):
        if set_type == 'train':
            examples = self._create_examples(self._read_json2(self.train_file), 'train')
            # 使用 open data + 比赛训练数据直接训练
            # examples = self._create_examples(self._read_json(self.train_file) + self._read_json(self.dev_file) * 2, 'train')
            examples = examples[:self.data_args.max_train_samples]
            self.train_examples = examples
        elif set_type == 'dev':
            examples = self._create_examples(self._read_json2(self.dev_file), 'dev')
            examples = examples[:self.data_args.max_eval_samples]
            self.dev_examples = examples
        elif set_type == 'test':
            examples = self._create_examples(self._read_json2(self.test_file), 'test')
            examples = examples[:self.data_args.max_predict_samples]
            self.test_examples = examples
        return examples

    def _create_examples(self, lines, set_type):
        '''
        train/dev:
        {
            "text": "而且还计划将民生银行作为战略投资者引入进来。",
            "label": {"company": {"民生银行": [[6, 9]]}, "position": {"战略投资者": [[12, 16]]}}
        }
        test:
        {
            "id": 1,
            "text": "而且还计划将民生银行作为战略投资者引入进来。"
        }
        '''
        examples = []
        is_train = 0 if set_type == 'test' else 1
        num = 0
        for line in lines:
            id_: int = line['id'] if 'id' in line.keys() else num # 原始数据的编号
            text: str = line['text'] # 原始文本
            target: dict = line['label'] if 'label' in line.keys() else dict()
            for label, label_word in self.verbalizer.items():
                desc = self.label_desc[label]
                start, end = list(), list()
                prompt_text = self.prompt[self.prompt_id].format(desc, label_word, text)
                new_len = len(self.prompt[self.prompt_id]) - 7 + len(desc) + len(label_word) # 新增的token数量
                if label in target.keys():
                    entity2spans = target[label]
                    target_ents = list()
                    for entity, spans in entity2spans.items():
                        target_ents.append(entity)
                        for s, e in spans:
                            start.append(s + new_len)
                            end.append(e + new_len)
                    target_ents = '|'.join(target_ents)
                else:
                    target_ents = ""
                    start, end = [-1], [-1]
                    # if set_type == 'train' and random.random() < 0.9:
                    #     continue
                # if num % 10 == 0 and is_train == 1:
                #     print("content=", prompt_text)
                #     print("start=", start)
                #     print("end=", end)
                #     print("label=", label)
                #     print("target_ents=", target_ents)
                #     print("is_train=", is_train)
                #     print("="*20)
                examples.append(
                    {
                        'id': id_,
                        'text': text,
                        'content': prompt_text,
                        'start': start,
                        'end': end,
                        'type': label,
                        'target': target_ents,
                        'data_type': 'clue_ner',
                        'is_train': is_train
                    }
                )
            num += 1

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
        # print('probs=', probs) # [n, m]
        # print('indices=', indices) # [n, m]
        predictions = {}
        topk_predictions = {}
        num = 0
        for prob, index, example in zip(probs, indices, examples):
            # new_len = len(self.prompt[self.prompt_id]) - 5 + len(self.verbalizer[example['type']])  # 新增的token数量
            new_len = len(self.prompt[self.prompt_id]) - 7 + len(self.label_desc[example['type']]) + len(self.verbalizer[example['type']])
            data_type = example['data_type']
            type = example['type']
            id_ = example['id'] # 因为有10个类别的实体，因此具有相同id的样本将对应10个样本
            index_ids = torch.Tensor([i for i in range(len(index))]).long()
            if id_ not in predictions.keys():
                predictions[id_] = dict()
            if id_ not in topk_predictions.keys():
                topk_predictions[id_] = dict()
            answer = []
            answer_check = []
            # ans2idx = dict()
            topk_answer = []
            # TODO 1. 调节阈值 2. 处理输出实体重叠问题
            entity_index = index[prob > 0.05]
            index_ids = index_ids[prob > 0.05]
            for ei, entity in enumerate(entity_index):
                num += 1
                # 1D index转2D index
                pred_prob = float(prob[index_ids[ei]])
                start_end = np.unravel_index(entity, (self.data_args.max_seq_length, self.data_args.max_seq_length))
                # print("start_end=", start_end)
                s = example['offset_mapping'][start_end[0]][0] - 1
                e = example['offset_mapping'][start_end[1]][1] - 1
                # print("s={}, e={}".format(s, e))
                if s == -1 and e == -1:
                    ans = ""
                else:
                    # post process 1: 如果预测的区间不在原始文本上，需要进行删减
                    if s - new_len < 0 and e - new_len >= 0:
                        s = new_len
                        ans = example['content'][s: e]
                        # print(ans)
                    elif s - new_len < 0 and e - new_len < 0:
                        s, e = -1, -1
                        ans = ""
                    else:
                        ans = example['content'][s: e]

                    # post process 2: clue ner的特例：如果预测的结果中，除了第一个字符是中文外，其余都是非中文，则删除该中文
                    if len(ans) > 1:
                        if _is_chinese_char(ans[0]) and not _is_chinese_char(ans[1]):
                            ans = ans[1:]
                            s = s + 1
                    # post process 3：如果预测的答案存在下面几种case，则删除最后一个字符
                    # 《哈利波特》{
                    # 吉普森阶梯g
                    if len(ans) > 1:
                        if ans[-2] == "》" or (_is_chinese_char(ans[-2]) and _is_eng_or_pun(ans[-1])):
                            ans = ans[:-1]
                            e = e - 1
                    # post process 4: 头尾是引号
                    if len(ans) > 1:
                        if ans[0] == "“" and ans[-1] == "”":
                            ans = ans[1:-1]
                            s = s + 1
                            e = e - 1
                        # if ans[0] == "“":
                        #     ans = ans[1:]
                        #     s = s + 1
                        # if ans[-1] == "”":
                        #     ans = ans[:-1]
                        #     e = e - 1
                # if num % 10 == 0:
                #     print("ans=", ans)
                    # print("*" * 20)
                if ans not in answer_check:
                    if pred_prob >= 0.2:
                        # ans2idx[ans] = len(answer)
                        answer.append({'answer': ans, 'prob': pred_prob, 'pos': [s, e]})
                        answer_check.append(ans)
                    topk_answer.append({'answer': ans, 'prob': pred_prob, 'pos': [s, e]})

            predictions[id_][type] = answer
            topk_predictions[id_][type] = topk_answer

        return predictions, topk_predictions

    def compute_metrics(self, eval_predictions):
        examples = self.raw_datasets['validation']
        golden, preds, dataname_map, dataname_type = {}, {}, defaultdict(list), {}
        predictions, _ = self.get_predict_result(eval_predictions[0], examples)
        self.save_result(eval_predictions[0], None, stage='dev')
        for example in examples:
            data_type = example['data_type']
            # dataname = "clue_ner"
            # if dataname not in dataname_type:
            #     dataname_type[dataname] = data_type
            id_ = example['id']
            label = example['type']
            golden_ = example['target'].split('|')
            preds_ = [i['answer'] for i in predictions[id_][label] if i['answer'] != ""]
            if len(preds_) == 0:
                preds_ = [""]
            if len(golden_) == 1 and len(preds_) == 1 and golden_[0] == "" and preds_[0] == "":
                continue

            golden["{}_{}".format(id_, label)] = golden_
            # 样本id对应label类别的预测实体
            preds["{}_{}".format(id_, label)] = preds_
            # print("label={}, golden={}".format(label, golden_))
            # print("label={}, preds={}".format(label, preds_))
            # print("*" * 20)
            # dataname_map[dataname].append(id_)



        all_metrics = {
            "macro_f1": 0.,
            "micro_f1": 0.,
            "eval_num": 0,
        }

        # for dataname, data_ids in dataname_map.items():
        metric = datatype2metrics["clue_ner"]()
        gold = {k: v for k, v in golden.items()}
        pred = {k: v for k, v in preds.items()}
        # pred = {"dev-{}".format(value['id']): value['label'] for value in predictions if "dev-{}".format(value['id']) in data_ids}
        score = metric.calc_metric(golden=gold, predictions=pred)
        acc, f1 = score['acc'], score['f1']
        # if len(gold) != len(pred) or len(gold) < 20:
        #     print(dataname, dataname_type[dataname], round(score, 4), len(gold), len(pred), data_ids)
        all_metrics["macro_f1"] = f1
        # all_metrics["micro_f1"] = score * len(gold)
        all_metrics["eval_num"] = len(gold)
        # all_metrics[dataname] = round(score, 4)
        return all_metrics

    def save_result(self, logits, label_ids, stage='test'):

        def search_all_span(entity, text):
            # 找到句子中所有实体的索引区间
            # spans = [[r.span()[0], r.span()[1] - 1] for r in re.finditer(entity, text)]
            if entity == "":
                return list()
            spans = [[i, i + len(entity) - 1] for i in range(len(text)) if text[i: i + len(entity)] == entity]
            return spans

        def remove_nested_span(ent_spans):
            # e.g. ent_spans = {"金融街": [[23, 25]], "北京的市中心": [[12, 17]], "北京": [[12, 13]], "市中心": [[15, 17]]}
            span_list = list()
            new_res = dict()
            # 先保存所有span
            for ent, span in ent_spans.items():
                span_list.extend(span)
            # 遍历每个实体，判断其是否与某一个实体存在嵌套
            for ent, spans in ent_spans.items():
                new_spans = list()
                for span in spans:
                    yes = False
                    for other_span in span_list:
                        if span == other_span: # 排除自己的区间
                            continue
                        if span[0] >= other_span[0] and span[1] <= other_span[1]: # 存在嵌套，当前的实体span被其他span包含
                            yes = True
                            break
                    if not yes: # 只有不存在嵌套的span才保留下
                        new_spans.append(span)
                if len(new_spans) != 0:
                    new_res[ent] = new_spans
            return new_res

        examples = self.raw_datasets['test'] if stage == 'test' else self.raw_datasets['validation']
        predicts, topk_predictions = self.get_predict_result(logits, examples)
        # clue_processor = clue_processors[self.data_name]()
        # label2word = clue_processor.get_verbalizers()
        # word2label = {v: k for k, v in label2word.items()}

        exampleid2index = dict()
        for ei, example in enumerate(examples):
            id_ = example["id"]
            exampleid2index[id_] = ei

        ### submit 格式转换为clue的
        answers = list()
        for k, v in predicts.items():
            # k: example id
            # {
            #     "text": "而且还计划将民生银行作为战略投资者引入进来。",
            #     "label": {"company": {"民生银行": [[6, 9]]}, "position": {"战略投资者": [[12, 16]]}}
            # }
            res = dict()
            for type, answer in v.items():
                # answer: [{'answer': ans, 'prob': pred_prob, 'pos': (s, e)}, ...]
                if len(answer) == 0:
                    # 说明当前类没有预测的实体
                    continue
                type_res = dict()
                if type not in res.keys():
                    res[type] = dict()
                answer_num = 0
                for ans in answer:
                    entity = ans['answer']
                    if entity == "":
                        continue
                    # span = ans['pos']
                    answer_num += 1
                    type_res[entity] = search_all_span(entity, examples[exampleid2index[k]]["text"])
                if answer_num > 0:
                    res[type] = type_res

            res_new = dict()
            for type, ent_spans in res.items():
                # 去除空答案
                if ent_spans == {}:
                    continue
                # post process: 同类的所有预测实体区间，对存在区间重叠的进行处理：
                # 1。如果区间相邻，则合并；2。如果区间包含关系，则取最大；3。区间不包含但有交集，则取prob最大
                # ent_spans {"金融街": [[23, 25]], "北京的市中心": [[12, 17]], "北京": [[12, 13]], "市中心": [[15, 17]]}
                ent_spans = remove_nested_span(ent_spans)
                res_new[type] = ent_spans

            answers.append({"id": k, "label": res_new})

        # outfile = os.path.join(self.training_args.output_dir, 'answer.json')
        # with open(outfile, 'w', encoding='utf-8') as f:
        # #     json.dump(predicts, f, ensure_ascii=False, indent=2)
        #     for res in answers:
        #         f.write("{}\n".format(str(res)))

        output_submit_file = os.path.join(
            self.training_args.output_dir,
            "cluener_predict.json" if stage == 'test' else "cluener_dev_predict.json")
        # 保存标签结果
        with open(output_submit_file, "w", encoding="utf-8") as writer:
            for i, pred in enumerate(answers):
                json_d = {}
                json_d['id'] = i
                json_d['label'] = pred["label"]
                writer.write(json.dumps(json_d, ensure_ascii=False) + '\n')

        if stage == 'test':
            # 保存TopK个预测结果
            topfile = os.path.join(self.training_args.output_dir, 'top20_predict.json')
            with open(topfile, 'w', encoding='utf-8') as f2:
                json.dump(topk_predictions, f2, ensure_ascii=False, indent=4)

