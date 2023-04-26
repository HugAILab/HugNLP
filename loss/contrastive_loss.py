# -*- coding: utf-8 -*-
# @Time    : 2022/03/23 14:50
# @Author  : Jianing Wang
# @Email   : lygwjn@gmail.com
# @File    : ContrastiveLoss.py
# !/usr/bin/env python
# coding=utf-8

from enum import Enum
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers.models.bert.modeling_bert import BertModel
from transformers import BertTokenizer, BertConfig

class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1-F.cosine_similarity(x, y)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.

    @:param distance_metric: The distance metric function
    @:param margin: (float) The margin distance
    @:param size_average: (bool) Whether to get averaged loss

    Input example of forward function:
        rep_anchor: [[0.2, -0.1, ..., 0.6], [0.2, -0.1, ..., 0.6], ..., [0.2, -0.1, ..., 0.6]]
        rep_candidate: [[0.3, 0.1, ...m -0.3], [-0.8, 1.2, ..., 0.7], ..., [-0.9, 0.1, ..., 0.4]]
        label: [0, 1, ..., 1]

    Return example of forward function:
        0.015 (averged)
        2.672 (sum)
    """

    def __init__(self, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.5, size_average:bool = False):
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.size_average = size_average

    def forward(self, rep_anchor, rep_candidate, label: Tensor):
        # rep_anchor: [batch_size, hidden_dim] denotes the representations of anchors
        # rep_candidate: [batch_size, hidden_dim] denotes the representations of positive / negative
        # label: [batch_size, hidden_dim] denotes the label of each anchor - candidate pair

        distances = self.distance_metric(rep_anchor, rep_candidate)
        losses = 0.5 * (label.float() * distances.pow(2) + (1 - label).float() * F.relu(self.margin - distances).pow(2))
        return losses.mean() if self.size_average else losses.sum()


if __name__ == "__main__":
    # configure for huggingface pre-trained language models
    config = BertConfig.from_pretrained("bert-base-cased")
    # tokenizer for huggingface pre-trained language models
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # pytorch_model.bin for huggingface pre-trained language models
    model = BertModel.from_pretrained("bert-base-cased")
    # obtain two batch of examples, each corresponding example is a pair
    examples1 = ["This is the sentence anchor 1.", "It is the second sentence in this article named Section D."]
    examples2 = ["It is the same as anchor 1.", "I think it is different with Section D."]
    label = [1, 0]
    # convert each example for feature
    # {"input_ids": xxx, "attention_mask": xxx, "token_tuype_ids": xxx}
    features1 = tokenizer(examples1, add_special_tokens=True, padding=True)
    features2 = tokenizer(examples2, add_special_tokens=True, padding=True)
    # padding and convert to feature batch
    max_seq_lem = 16
    features1 = {key: torch.Tensor([value + [0] * (max_seq_lem - len(value)) for value in values]).long() for key, values in features1.items()}
    features2 = {key: torch.Tensor([value + [0] * (max_seq_lem - len(value)) for value in values]).long() for key, values in features2.items()}
    label = torch.Tensor(label).long()
    # obtain sentence embedding by averaged pooling
    rep_anchor = model(**features1)[0] # [batch_size, max_seq_len, hidden_dim]
    rep_candidate = model(**features2)[0] # [batch_size, max_seq_len, hidden_dim]
    rep_anchor = torch.mean(rep_anchor, -1) # [batch_size, hidden_dim]
    rep_candidate = torch.mean(rep_candidate, -1) # [batch_size, hidden_dim]
    # obtain contrastive loss
    loss_fn = ContrastiveLoss()
    loss = loss_fn(rep_anchor=rep_anchor, rep_candidate=rep_candidate, label=label)
    print(loss) # tensor(0.0869, grad_fn=<SumBackward0>)
