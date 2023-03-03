# -*- coding: utf-8 -*-
# @Time    : 2022/03/23 15:25
# @Author  : Jianing Wang
# @Email   : lygwjn@gmail.com
# @File    : TripletLoss.py
# !/usr/bin/env python
# coding=utf-8

from enum import Enum
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel
from transformers import BertTokenizer, BertConfig

class TripletDistanceMetric(Enum):
    """
    The metric for the triplet loss
    """
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)

class TripletLoss(nn.Module):
    """
    This class implements triplet loss. Given a triplet of (anchor, positive, negative),
    the loss minimizes the distance between anchor and positive while it maximizes the distance
    between anchor and negative. It compute the following loss function:

    loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0).

    Margin is an important hyperparameter and needs to be tuned respectively.

    @:param distance_metric: The distance metric function
    @:param triplet_margin: (float) The margin distance

    Input example of forward function:
        rep_anchor: [[0.2, -0.1, ..., 0.6], [0.2, -0.1, ..., 0.6], ..., [0.2, -0.1, ..., 0.6]]
        rep_candidate: [[0.3, 0.1, ...m -0.3], [-0.8, 1.2, ..., 0.7], ..., [-0.9, 0.1, ..., 0.4]]
        label: [0, 1, ..., 1]

    Return example of forward function:
        0.015 (averged)
        2.672 (sum)

    """
    def __init__(self, distance_metric=TripletDistanceMetric.EUCLIDEAN, triplet_margin: float = 0.5):
        super(TripletLoss, self).__init__()
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin


    def forward(self, rep_anchor, rep_positive, rep_negative):
        # rep_anchor: [batch_size, hidden_dim] denotes the representations of anchors
        # rep_positive: [batch_size, hidden_dim] denotes the representations of positive, sometimes, it canbe dropout
        # rep_negative: [batch_size, hidden_dim] denotes the representations of negative
        # label: [batch_size, hidden_dim] denotes the label of each anchor - candidate pair
        distance_pos = self.distance_metric(rep_anchor, rep_positive)
        distance_neg = self.distance_metric(rep_anchor, rep_negative)

        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
        return losses.mean()


if __name__ == "__main__":
    # configure for huggingface pre-trained language models
    config = BertConfig.from_pretrained("bert-base-cased")
    # tokenizer for huggingface pre-trained language models
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # pytorch_model.bin for huggingface pre-trained language models
    model = BertModel.from_pretrained("bert-base-cased")
    # obtain two batch of examples, each corresponding example is a pair
    anchor_example = ["I am an anchor, which is the source example sampled from corpora."] # anchor sentence
    positive_example = [
        "I am an anchor, which is the source example.",
        "I am the source example sampled from corpora."
    ] # positive, which randomly dropout or noise from anchor
    negative_example = [
        "It is different with the anchor.",
        "My name is Jianing Wang, please give me some stars, thank you!"
    ] # negative, which randomly sampled from corpora
    # convert each example for feature
    # {"input_ids": xxx, "attention_mask": xxx, "token_tuype_ids": xxx}
    anchor_feature = tokenizer(anchor_example, add_special_tokens=True, padding=True)
    positive_feature = tokenizer(positive_example, add_special_tokens=True, padding=True)
    negative_feature = tokenizer(negative_example, add_special_tokens=True, padding=True)
    # padding and convert to feature batch
    max_seq_lem = 24
    anchor_feature = {key: torch.Tensor([value + [0] * (max_seq_lem - len(value)) for value in values]).long() for key, values in anchor_feature.items()}
    positive_feature = {key: torch.Tensor([value + [0] * (max_seq_lem - len(value)) for value in values]).long() for key, values in positive_feature.items()}
    negative_feature = {key: torch.Tensor([value + [0] * (max_seq_lem - len(value)) for value in values]).long() for key, values in negative_feature.items()}
    # obtain sentence embedding by averaged pooling
    rep_anchor = model(**anchor_feature)[0] # [1, max_seq_len, hidden_dim]
    rep_positive = model(**positive_feature)[0] # [batch_size, max_seq_len, hidden_dim]
    rep_negative = model(**negative_feature)[0] # [batch_size, max_seq_len, hidden_dim]
    # repeat
    rep_anchor = torch.mean(rep_anchor, -1) # [1, hidden_dim]
    rep_positive = torch.mean(rep_positive, -1) # [batch_size, hidden_dim]
    rep_negative = torch.mean(rep_negative, -1) # [batch_size, hidden_dim]
    # obtain contrastive loss
    loss_fn = TripletLoss()
    loss = loss_fn(rep_anchor=rep_anchor, rep_positive=rep_positive, rep_negative=rep_negative)
    print(loss) # tensor(0.5001, grad_fn=<MeanBackward0>)
