# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 10:19 下午
# @Author  : JianingWang
# @File    : data_collator.py
import torch
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
from transformers import PreTrainedTokenizerBase



@dataclass
class DataCollatorForTokenProto:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 64
    num_class: Optional[int] = None
    num_example: Optional[int] = 5
    mode: Optional[str] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_max_length: Optional[bool] = None
    path: Optional[bool] = None

    def __call__(self, features):
        batch_support = {'word': [], 'mask': [], 'label': [], 'sentence_num':[], 'text_mask':[]}
        batch_query = {'word': [], 'mask': [], 'label': [], 'sentence_num':[], 'label2tag':[], 'text_mask':[]}
        # support_sets, query_sets = zip(*features)

        for feature_id, feature in enumerate(features): # 遍历每一个episode
            batch_support["word"].extend(feature["support_word"])
            batch_support["mask"].extend(feature["support_mask"])
            batch_support["label"].extend(feature["support_label"])
            batch_support["sentence_num"].extend(feature["support_sentence_num"])
            batch_support["text_mask"].extend(feature["support_text_mask"])

            batch_query["word"].extend(feature["query_word"])
            batch_query["mask"].extend(feature["query_mask"])
            batch_query["label"].extend(feature["query_label"])
            batch_query["sentence_num"].extend(feature["query_sentence_num"])
            batch_query["text_mask"].extend(feature["query_text_mask"])

        batch_support = {keys: torch.tensor(values) for keys, values in batch_support.items()}
        batch_query = {keys: torch.tensor(values) for keys, values in batch_query.items()}
        # for i in range(len(support_sets)):
        #     for k in batch_support:
        #         batch_support[k] += support_sets[i][k]
        #     for k in batch_query:
        #         batch_query[k] += query_sets[i][k]
        # for k in batch_support:
        #     if k != 'label' and k != 'sentence_num':
        #         batch_support[k] = torch.stack(batch_support[k], 0)
        # for k in batch_query:
        #     if k !='label' and k != 'sentence_num' and k!= 'label2tag':
        #         batch_query[k] = torch.stack(batch_query[k], 0)
        # batch_support['label'] = [torch.tensor(tag_list).long() for tag_list in batch_support['label']]
        # batch_query['label'] = [torch.tensor(tag_list).long() for tag_list in batch_query['label']]
        return batch_support, batch_query




def generate_global_pointer_labels(data_set: dict, max_length):
    # data_set: support/query set
    # 获得support或query set对应的global pointer的label
    labeled_spans, labeled_types, offset_mappings = data_set['labeled_spans'], data_set['labeled_types'], data_set['offset_mapping']
    new_labeled_spans, new_labeled_types = list(), list()
    labels = torch.zeros(
        len(labeled_spans), 1, max_length, max_length
    )  # 阅读理解任务entity种类为1 [bz, 1, max_len, max_len]


    for ei in range(len(labeled_spans)): # 遍历每一个句子
        labeled_span = labeled_spans[ei] # list # 当前句子的所有mention span（字符级别）
        labeled_type = labeled_types[ei]
        offset = offset_mappings[ei]  # 表示tokenizer生成的token对应原始文本中字符级别的位置区间
        new_labeled_span, new_labeled_type = list(), list() # 当前句子的所有mention span（token级别）
        position_map = {}
        for i, (m, n) in enumerate(offset): # 第i个分词对应原始文本中字符级别的区间(m, n)
            if i != 0 and m == 0 and n == 0:
                continue
            for k in range(m, n + 1):
                position_map[k] = i # 字符级别的第k个字符属于分词i
        if len(labeled_span) == 0:
            labels[ei, 0, 0, 0] = 1
            new_labeled_span.append([])
            new_labeled_types.append([])
        for ej, span in enumerate(labeled_span): # 遍历每个span
            start, end = span
            end -= 1
            # MRC 没有答案时则把label指向CLS
            # if start == 0:
            #     # assert end == -1
            #     labels[ei, 0, 0, 0] = 1
            #     new_labeled_span.append([0, 0])

            if start in position_map and end in position_map:
                # 指定下列元素为1，说明表示第feature_id个样本的预测区间
                labels[ei, 0, position_map[start], position_map[end]] = 1
                new_labeled_span.append([position_map[start], position_map[end]])
                new_labeled_type.append(labeled_type[ej])
        new_labeled_spans.append(new_labeled_span)
        new_labeled_types.append(new_labeled_type)
    return labels, new_labeled_spans, new_labeled_types



@dataclass
class DataCollatorForSpanProto:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 64
    num_class: Optional[int] = None
    num_example: Optional[int] = 5
    mode: Optional[str] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_max_length: Optional[bool] = None
    path: Optional[bool] = None

    def __call__(self, features):
        '''
            由dataloader随机采样的一个batch
            input:
            features = [
                {
                    'id': xx
                    'support_input': {
                        'input_ids': [[xxx], ...],
                        'attention_mask': [[xxx], ...],
                        'token_type_ids': [[xxx], ...],
                        'offset_mapping': xxxx
                    },
                    'query_input': {
                        'input_ids': [[xxx], ...],
                        'attention_mask': [[xxx], ...],
                        'token_type_ids': [[xxx], ...],
                        'offset_mapping': xxx
                    },
                    'support_labeled_spans': [[[x, x], ..], ..],
                    'support_labeled_types': [[xx, ..], ..],
                    'support_sentence_num': xx,
                    'query_labeled_spans': [[[x, x], ..], ..],
                    'query_labeled_types': [[xx, ..], ..],
                    'query_sentence_num': xx,
                    'stage': xx,
                }
            ]

            return
            features = {
                'support': {
                    'input_ids': [],
                    'attention_mask': [[xxx], ...],
                    'token_type_ids': [[xxx], ...],
                    'labeled_spans':,
                    'labeled_types':,
                    'sentence_num': [xx, ...],
                    'labels': []
                },
                'query': {
                    'input_ids': [],
                    'attention_mask': [[xxx], ...],
                    'token_type_ids': [[xxx], ...],
                    'labeled_spans':,
                    'labeled_types':,
                    'sentence_num':,
                    'labels': []
                },
                'num_class': x,
            }
        '''
        id_batch = list()
        support_batch = {
            'input_ids': list(), 'attention_mask': list(), 'token_type_ids': list(), 'offset_mapping': list(),
            'labeled_spans': list(), 'labeled_types': list(), 'sentence_num': list(), 'labels': list()
        }
        query_batch = {
            'input_ids': list(), 'attention_mask': list(), 'token_type_ids': list(), 'offset_mapping': list(),
            'labeled_spans': list(), 'labeled_types': list(), 'sentence_num': list(), 'labels': list()
        }

        # all_support_sentence_num, all_query_sentence_num = 0, 0
        stage = features[0]['stage']

        # if stage == "dev":
        #     print('0 collator stage=', stage)
        for feature_id, feature in enumerate(features): # 遍历每一个episode
            # print(feature)
            # 获得每个episode的support和query对应的输入句子、attention等三件套，并进行padding
            id_batch.append(feature['id'])
            if 'num_class' in feature.keys():
                self.num_class = feature['num_class']
                # print('self.num_class', self.num_class)
            support_input, query_input = feature['support_input'], feature['query_input']
            support_input = self.tokenizer.pad(
                support_input,
                padding='max_length',  # 为了index不出错直接Padding到max length，如果用longest，后面的np.unravel_index也要改
                max_length=self.max_length,
            )

            query_input = self.tokenizer.pad(
                query_input,
                padding='max_length',  # 为了index不出错直接Padding到max length，如果用longest，后面的np.unravel_index也要改
                max_length=self.max_length,
            )

            # 将整个batch内的所有episode的所有输入进行合并
            support_batch['input_ids'].extend(support_input['input_ids'])
            support_batch['attention_mask'].extend(support_input['attention_mask'])
            support_batch['token_type_ids'].extend(support_input['token_type_ids'])
            support_batch['offset_mapping'].extend(support_input['offset_mapping'])

            query_batch['input_ids'].extend(query_input['input_ids'])
            query_batch['attention_mask'].extend(query_input['attention_mask'])
            query_batch['token_type_ids'].extend(query_input['token_type_ids'])
            query_batch['offset_mapping'].extend(query_input['offset_mapping'])


            # 其他span以及span_type的合并
            support_batch['labeled_spans'].extend(feature['support_labeled_spans'])
            support_batch['labeled_types'].extend(feature['support_labeled_types'])
            support_batch['sentence_num'].append(feature['support_sentence_num'])
            # all_support_sentence_num += feature['support_sentence_num'] # 记录当前一整个batch内所有episode的句子数量

            query_batch['labeled_spans'].extend(feature['query_labeled_spans'])
            query_batch['labeled_types'].extend(feature['query_labeled_types'])
            query_batch['sentence_num'].append(feature['query_sentence_num'])
            # all_query_sentence_num += feature['query_sentence_num'] # 记录当前一整个batch内所有episode的句子数量

        # support_labels = torch.zeros(
        #     all_support_sentence_num, 1, self.max_length, self.max_length
        # )  # 阅读理解任务entity种类为1 [bz, 1, max_len, max_len]
        # query_labels = torch.zeros(
        #     all_query_sentence_num, 1, self.max_length, self.max_length
        # )

        ############################

        # # 为global pointer提供label
        support_labels, support_new_labeled_spans, support_new_labeled_types = generate_global_pointer_labels(support_batch, self.max_length)
        query_labels, query_new_labeled_spans, query_new_labeled_types = generate_global_pointer_labels(query_batch, self.max_length)
        support_batch.pop('offset_mapping')
        query_batch.pop('offset_mapping')

        support_batch['labeled_spans'] = support_new_labeled_spans
        support_batch['labeled_types'] = support_new_labeled_types
        query_batch['labeled_spans'] = query_new_labeled_spans
        query_batch['labeled_types'] = query_new_labeled_types
        # print("input_ids=", support_batch['input_ids'][0])
        # print("labeled_span=", support_batch['labeled_spans'][0])
        # print("labeled_type=", support_batch['labeled_types'][0])

        # check
        # for span, type in zip(support_batch['labeled_spans'], support_batch['labeled_types']):
        #     assert len(span) == len(type), "support\nspan={}\ntype{}".format(span, type)
        #     if len(span) > 10:
        #         print("support\nspan={}\ntype{}".format(span, type))
        #
        # for span, type in zip(query_batch['labeled_spans'], query_batch['labeled_types']):
        #     assert len(span) == len(type), "query\nspan={}\ntype{}".format(span, type)
        #     if len(span) > 10:
        #         print("support\nspan={}\ntype{}".format(span, type))

        # short_labels没用，解决transformers trainer默认会merge labels导致内存爆炸的问题
        # 需配合--label_names=short_labels使用
        support_batch['labels'] = support_labels
        if support_batch['labels'].max() > 0:
            support_batch['short_labels'] = torch.ones(len(support_batch['labeled_spans']))
        else:
            support_batch['short_labels'] = torch.zeros(len(support_batch['labeled_spans']))

        query_batch['labels'] = query_labels
        if query_batch['labels'].max() > 0:
            query_batch['short_labels'] = torch.ones(len(query_batch['labeled_spans']))
        else:
            query_batch['short_labels'] = torch.zeros(len(query_batch['labeled_spans']))

        # convert to torch
        # id_batch = torch.Tensor(id_batch).long()
        support_batch['input_ids'] = torch.Tensor(support_batch['input_ids']).long()
        support_batch['attention_mask'] = torch.Tensor(support_batch['attention_mask']).long()
        support_batch['token_type_ids'] = torch.Tensor(support_batch['token_type_ids']).long()
        query_batch['input_ids'] = torch.Tensor(query_batch['input_ids']).long()
        query_batch['attention_mask'] = torch.Tensor(query_batch['attention_mask']).long()
        query_batch['token_type_ids'] = torch.Tensor(query_batch['token_type_ids']).long()

        # print('====')
        # if stage == "dev":
        #     print('1 collator stage=', stage)
        return {
            'episode_ids': id_batch,
            'support': support_batch,
            'query': query_batch,
            'num_class': self.num_class,
            'num_example': self.num_example,
            'mode': self.mode,
            'stage': stage,
            'short_labels': torch.zeros(len(features)),
            'path': self.path
        }
