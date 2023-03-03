# -*- coding: utf-8 -*-
# @Time    : 2022/03/23 16:55
# @Author  : Jianing Wang
# @Email   : lygwjn@gmail.com
# @File    : SimilarityLoss.py
# !/usr/bin/env python
# coding=utf-8

import torch
from torch import nn, Tensor
from transformers.models.bert.modeling_bert import BertModel
from transformers import BertTokenizer, BertConfig


class CosineSimilarityLoss(nn.Module):
    """
    CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.

    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.

    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).


    """
    def __init__(self, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation


    def forward(self, rep_a, rep_b, label: Tensor):
        # rep_a: [batch_size, hidden_dim]
        # rep_b: [batch_size, hidden_dim]
        output = self.cos_score_transformation(torch.cosine_similarity(rep_a, rep_b))
        # print(output) # tensor([0.9925, 0.5846], grad_fn=<DivBackward0>), tensor(0.1709, grad_fn=<MseLossBackward0>)
        return self.loss_fct(output, label.view(-1))



if __name__ == "__main__":
    # configure for huggingface pre-trained language models
    config = BertConfig.from_pretrained("bert-base-cased")
    # tokenizer for huggingface pre-trained language models
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # pytorch_model.bin for huggingface pre-trained language models
    model = BertModel.from_pretrained("bert-base-cased")
    # obtain two batch of examples, each corresponding example is a pair
    examples1 = ["Beijing is one of the biggest city in China.", "Disney film is well seeing for us."]
    examples2 = ["Shanghai is the largest city in east of China.", "ACL 2021 will be held in line due to COVID-19."]
    label = [1, 0]
    # convert each example for feature
    # {"input_ids": xxx, "attention_mask": xxx, "token_tuype_ids": xxx}
    features1 = tokenizer(examples1, add_special_tokens=True, padding=True)
    features2 = tokenizer(examples2, add_special_tokens=True, padding=True)
    # padding and convert to feature batch
    max_seq_lem = 24
    features1 = {key: torch.Tensor([value + [0] * (max_seq_lem - len(value)) for value in values]).long() for key, values in features1.items()}
    features2 = {key: torch.Tensor([value + [0] * (max_seq_lem - len(value)) for value in values]).long() for key, values in features2.items()}
    label = torch.Tensor(label).long()
    # obtain sentence embedding by averaged pooling
    rep_a = model(**features1)[0] # [batch_size, max_seq_len, hidden_dim]
    rep_b = model(**features2)[0] # [batch_size, max_seq_len, hidden_dim]
    rep_a = torch.mean(rep_a, -1)  # [batch_size, hidden_dim]
    rep_b = torch.mean(rep_b, -1)  # [batch_size, hidden_dim]
    # obtain contrastive loss
    loss_fn = CosineSimilarityLoss()
    loss = loss_fn(rep_a=rep_a, rep_b=rep_b, label=label)
    print(loss) # tensor(0.1709, grad_fn=<SumBackward0>)
