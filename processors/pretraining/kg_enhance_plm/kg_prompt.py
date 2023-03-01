# -*- coding: utf-8 -*-
# @Time    : 2022/3/16 13:30 下午
# @Author  : ruihan.wjn
# @File    : kg_prompt
import numpy as np
import os
from transformers import PreTrainedTokenizerBase
from typing import Dict
import random


class KGPrompt:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

        # 加载Wikidata5M知识库
        print('loading wikidata5m knowledge graph ...')
        kg_output = './pretrain_data/kg/'
        kg = np.load(os.path.join(kg_output, 'wiki_kg.npz'), allow_pickle=True)
        self.wiki5m_alias2qid, self.wiki5m_qid2alias, self.wiki5m_pid2alias, self.head_cluster = \
            kg['wiki5m_alias2qid'][()], kg['wiki5m_qid2alias'][()], kg['wiki5m_pid2alias'][()], kg['head_cluster'][()]
        print('loading success .')

    def sample_entity(self, qid=None, neg_num=0):
        # 给定一个qid，随机采样一个positive，以及若干负样本
        positive = None
        negative = list()
        if not qid and qid in self.wiki5m_qid2alias:
            positive = random.sample(self.wiki5m_qid2alias[qid], 1)[0]

        if neg_num > 0:
            negative_qid = random.sample(self.wiki5m_qid2alias.keys(), neg_num)
            for i in negative_qid:
                negative.append(random.sample(self.wiki5m_qid2alias[i], 1)[0])
        return positive, negative

    def sample_relation(self, pid=None, neg_num=0):
        # 给定一个pid，随机采样一个positive，以及若干负样本
        positive = None
        negative = list()
        if not pid and pid in self.wiki5m_pid2alias:
            positive = random.sample(self.wiki5m_pid2alias[pid], 1)[0]
        if neg_num > 0:
            negative_pid = random.sample(self.wiki5m_pid2alias.keys(), neg_num)
            for i in negative_pid:
                negative.append(random.sample(self.wiki5m_pid2alias[i], 1)[0])
        return positive, negative

    def encode_kg(self, kg_str_or_list, max_len):
        # 将实体/关系分词，并pad
        if type(kg_str_or_list) == str:
            kg_str_or_list = [kg_str_or_list]
        kg_input_ids = list()
        for kg_str in kg_str_or_list:
            kg_ids = self.tokenizer.encode(kg_str,
                                           add_special_tokens=False,
                                           max_length=max_len)
            kg_ids = [self.tokenizer.cls_token_id
                      ] + kg_ids[:max_len - 2] + [self.tokenizer.sep_token_id]
            kg_ids.extend([self.tokenizer.pad_token_id] *
                          (max_len - len(kg_ids)))
            kg_input_ids.append(kg_ids)
            # print('len(kg_ids)=', len(kg_ids))
        return kg_input_ids

    def get_demonstration(self,
                          example: Dict,
                          is_negative=False,
                          start_from_input=True):
        '''
        e.g.
        data = {
            'token_ids': tokens,
            'entity_qid': entity_ids,
            'entity_pos': mention_spans,
            'relation_pid': None,
            'relation_pos': None,
        }
        '''

        # 分词
        input_ids, entity_ids = example['token_ids'], example['entity_qid']
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [
            self.tokenizer.sep_token_id
        ]  # roberta: [0, x, ..., 2]
        # input_ids = self.tokenizer.encode(token_list, add_special_tokens=True)  # [101, x, ..., 102]
        type_id = 0
        token_type_ids = [type_id] * len(input_ids)
        start_length = len(input_ids) if start_from_input else 0
        entity_spans, relation_spans = list(), list()
        token_type_span = list()  # 每个type对应的区间
        token_type_span.append((0, len(input_ids)))
        if is_negative:
            # 如果是采样负样本，则随机从KG采样一些实体id
            entity_ids = random.sample(self.wiki5m_qid2alias.keys(),
                                       len(entity_ids))
        type_id = 1
        # 获得所有mention对齐的entity，根据entity_id随机采样KB里的entity
        for entity_id in entity_ids:
            if entity_id in self.wiki5m_qid2alias.keys(
            ) and entity_id in self.head_cluster.keys():
                entity_name_list = self.wiki5m_qid2alias[
                    entity_id]  # ['xx', 'xxx', ...]
                cluster_list = self.head_cluster[
                    entity_id]  # [(rel_id, q_id), ...]
                # 随机采样一个entity
                head_name = random.sample(entity_name_list, 1)[0]
                triple = random.sample(cluster_list, 1)[0]
                if triple[0] in self.wiki5m_pid2alias.keys(
                ) and triple[1] in self.wiki5m_qid2alias.keys():

                    relation_name = self.wiki5m_pid2alias[triple[0]]
                    tail_name = random.sample(self.wiki5m_qid2alias[triple[1]],
                                              1)[0]
                    template_tokens, entity_span, relation_span = self.template(
                        head=head_name,
                        relation=relation_name,
                        tail=tail_name,
                        type_id=random.randint(0, 2),
                        start_length=start_length)
                    if len(input_ids) + len(
                            template_tokens
                    ) >= self.tokenizer.model_max_length - 2:
                        break
                    start = len(input_ids)
                    input_ids.extend(template_tokens)
                    end = len(input_ids)
                    token_type_ids.extend([type_id] * len(template_tokens))
                    entity_spans.extend(entity_span)
                    relation_spans.extend(relation_span)
                    token_type_span.append((start, end))

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'noise_detect_label': 0 if is_negative else 1,
            'entity_spans': entity_spans,
            'relation_spans': relation_spans,
            'token_type_span': token_type_span
        }

    def template(self, head, relation, tail, type_id=0, start_length=0):
        if type_id == 0:
            templates = [
                'The relation between', head, 'and', tail, 'is', relation
            ]
            flag = [0, 1, 0, 1, 0, 2]
        elif type_id == 1:
            templates = [head, relation, tail]
            flag = [1, 2, 1]
        elif type_id == 2:
            templates = [head, 'is the', relation, 'of', tail]
            flag = [1, 0, 2, 0, 1]

        template_tokens = list()
        entity_spans = list()
        relation_spans = list()
        for ei, string in enumerate(templates):
            start = start_length + len(template_tokens)
            tokens = self.tokenizer.encode(string, add_special_tokens=False)
            template_tokens.extend(tokens)
            end = start_length + len(template_tokens)
            if flag[ei] == 1:
                entity_spans.append((start, end))
            elif flag[ei] == 2:
                relation_spans.append((start, end))
        template_tokens += [self.tokenizer.sep_token_id]
        return template_tokens, entity_spans, relation_spans
