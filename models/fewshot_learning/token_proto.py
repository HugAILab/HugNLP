# -*- coding: utf-8 -*-
# @Time    : 2022/4/21 5:30 下午
# @Author  : JianingWang
# @File    : span_proto.py
"""This code is implemented for the paper ''SpanProto: A Two-stage Span-based Prototypical Network for Few-shot Named Entity Recognition''."""

import os
from typing import Optional
import torch
import numpy as np
import torch.nn as nn
from typing import Union
from dataclasses import dataclass
from torch.nn import BCEWithLogitsLoss
from transformers import MegatronBertModel, MegatronBertPreTrainedModel
from transformers.file_utils import ModelOutput
from transformers.models.bert import BertPreTrainedModel, BertModel


@dataclass
class TokenProtoOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class TokenProto(nn.Module):
    def __init__(self, config):
        '''
        word_encoder: Sentence encoder

        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.config = config
        self.output_dir = './outputs'
        # self.predict_dir = self.predict_result_path(self.output_dir)
        self.drop = nn.Dropout()
        self.projector = nn.Sequential(  # projector
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Sigmoid(),
            # nn.LayerNorm(2)
        )
        self.tag_embeddings = nn.Embedding(
            2, self.config.hidden_size)  # tag for labeled / unlabeled span set
        # self.tag_mlp = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.max_length = 64
        self.margin_distance = 6.0
        self.global_step = 0

    def predict_result_path(self, path=None):
        if path is None:
            predict_dir = os.path.join(
                self.output_dir, '{}-{}-{}'.format(self.mode, self.num_class,
                                                   self.num_example),
                'predict')
        else:
            predict_dir = os.path.join(path, 'predict')
        # if os.path.exists(predict_dir):
        #     os.rmdir(predict_dir) # 删除历史记录
        if not os.path.exists(predict_dir):  # 重新创建一个新的目录
            os.makedirs(predict_dir)
        return predict_dir

    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path: Optional[Union[str,
                                                               os.PathLike]],
            *model_args, **kwargs):
        config = kwargs.pop('config', None)
        model = TokenProto(config=config)
        return model

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q, q_mask):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        assert Q.size()[:2] == q_mask.size()
        Q = Q[q_mask == 1].view(
            -1, Q.size(-1))  # [num_of_all_text_tokens, embed_dim]
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def __get_proto__(self, embedding, tag, mask):
        proto = []
        embedding = embedding[mask == 1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embedding.size(0)
        for label in range(torch.max(tag) + 1):
            proto.append(torch.mean(embedding[tag == label], 0))
        proto = torch.stack(proto)
        return proto, embedding

    def forward(self, support, query):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set

        support/query = {'index': [], 'word': [], 'mask': [], 'label': [], 'sentence_num': [], 'text_mask': []}

        '''
        # support set和query set分别喂入BERT中获得各个样本的表示
        support_emb = self.word_encoder(
            support['word'],
            support['mask'])  # [num_sent, number_of_tokens, 768]
        query_emb = self.word_encoder(
            query['word'], query['mask'])  # [num_sent, number_of_tokens, 768]
        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)

        # Prototypical Networks
        logits = []
        current_support_num = 0
        current_query_num = 0
        assert support_emb.size()[:2] == support['mask'].size()
        assert query_emb.size()[:2] == query['mask'].size()

        for i, sent_support_num in enumerate(
                support['sentence_num']):  # 遍历每个采样得到的N-way K-shot任务数据
            sent_query_num = query['sentence_num'][i]
            # Calculate prototype for each class
            # 因为一个batch里对应多个episode，因此 current_support_num:current_support_num+sent_support_num
            # 用来表示当前输入的张量中，哪个范围内的句子属于当前N-way K-shot采样数据
            support_proto, embedding = self.__get_proto__(
                support_emb[current_support_num:current_support_num +
                            sent_support_num],
                support['label'][current_support_num:current_support_num +
                                 sent_support_num],
                support['text_mask'][current_support_num:current_support_num +
                                     sent_support_num])
            # calculate distance to each prototype
            logits.append(
                self.__batch_dist__(
                    support_proto,
                    query_emb[current_query_num:current_query_num +
                              sent_query_num], query['text_mask']
                    [current_query_num:current_query_num +
                     sent_query_num]))  # [num_of_query_tokens, class_num]
            current_query_num += sent_query_num
            current_support_num += sent_support_num
        logits = torch.cat(logits, 0)  # 每个query的从属于support set对应各个类的概率
        _, pred = torch.max(logits, 1)  # 挑选最大概率对应的proto类作为预测结果

        # return logits, pred, embedding

        return TokenProtoOutput(
            logits=logits
        )  # 返回部分的所有logits不论最外层是list还是tuple，最里层一定要包含一个张量，否则huggingface里的nested_detach函数会报错
