# -*- coding: utf-8 -*-
# @Time    : 2022/3/15 21:26
# @Author  : ruihan.wjn
# @File    : pk-plm.py

"""
This code is implemented for the paper ""Knowledge Prompting in Pre-trained Langauge Models for Natural Langauge Understanding""
"""

from time import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import OrderedDict
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.models.roberta import RobertaModel, RobertaPreTrainedModel, RobertaTokenizer, RobertaForMaskedLM
from transformers.models.deberta import DebertaModel, DebertaPreTrainedModel, DebertaTokenizer, DebertaForMaskedLM
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertPreTrainingHeads
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaLMHead
from transformers.models.deberta.modeling_deberta import DebertaModel, DebertaLMPredictionHead

"""
kg enhanced corpus structure example:
{
    "token_ids": [20, 46098, 3277, 680, 10, 4066, 278, 9, 11129, 4063, 877, 579, 8, 8750, 14720, 8, 22498, 548,
    19231, 46098, 3277, 6, 25, 157, 25, 130, 3753, 46098, 3277, 4, 3684, 19809, 10960, 9, 5, 30731, 2788, 914, 5,
    1675, 8151, 35], "entity_pos": [[8, 11], [13, 15], [26, 27]],
    "entity_qid": ["Q17582", "Q231978", "Q427013"],
    "relation_pos": null,
    "relation_pid": null
}
"""


from enum import Enum
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
    Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two emeddings. The class SiameseDistanceMetric contains pre-defined metrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.
    Example::
        from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
        from sentence_transformers.readers import InputExample
        model = SentenceTransformer("distilbert-base-nli-mean-tokens")
        train_examples = [InputExample(texts=["This is a positive pair", "Where the distance will be minimized"], label=1),
            InputExample(texts=["This is a negative pair", "Their distance will be increased"], label=0)]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.ContrastiveLoss(model=model)
    """

    def __init__(self, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.5, size_average:bool = True):
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.size_average = size_average

    def forward(self, sent_embs1, sent_embs2, labels: torch.Tensor):
        rep_anchor, rep_other = sent_embs1, sent_embs2
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        return losses.mean() if self.size_average else losses.sum()



class NSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score



class RoBertaKPPLMForProcessedWikiKGPLM(RobertaForMaskedLM):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # self.roberta = RobertaModel(config)
        try:
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
        except:
            classifier_dropout = (config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        # self.cls = BertOnlyMLMHead(config)
        # self.lm_head = RobertaLMHead(config)  # Masked Language Modeling head
        self.detector = NSPHead(config)  # Knowledge Noise Detection head
        self.entity_mlp = nn.Linear(config.hidden_size, config.hidden_size)
        self.relation_mlp = nn.Linear(config.hidden_size, config.hidden_size)
        # self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, config.num_ner_labels) for _ in range(config.entity_type_num)])

        self.contrastive_loss_fn = ContrastiveLoss()
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            # entity_label=None,
            entity_candidate=None,
            # relation_label=None,
            relation_candidate=None,
            noise_detect_label=None,
            task_id=None,
            mask_id=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # start_time = time()
        mlm_labels = labels

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print("attention_mask.shape=", attention_mask.shape)
        # print("input_ids[0]=", input_ids[0])
        # print("token_type_ids[0]=", token_type_ids[0])
        # attention_mask = None

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)  # mlm head
        # noise_detect_scores = self.detector(pooled_output)  # knowledge noise detector use pool output
        noise_detect_scores = self.detector(sequence_output[:, 0, :])  # knowledge noise detector use cls embedding

        # ner
        # sequence_output = self.dropout(sequence_output)
        # ner_logits = torch.stack([classifier(sequence_output) for classifier in self.classifiers]).movedim(1, 0)

        # mlm
        masked_lm_loss, noise_detect_loss, entity_loss, total_loss = None, None, None, None
        total_loss = list()
        if mlm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            total_loss.append(masked_lm_loss)

        # if noise_detect_label is not None:
        #     noise_detect_scores = noise_detect_scores[task_id == 1]
        #     noise_detect_label = noise_detect_label[task_id == 1]
        #
        #     if len(noise_detect_label) > 0:
        #         loss_fct = CrossEntropyLoss()
        #         noise_detect_loss = loss_fct(noise_detect_scores.view(-1, 2), noise_detect_label.view(-1))
        #         total_loss.append(noise_detect_loss)

        entity_candidate = entity_candidate[task_id == 2]
        if len(entity_candidate) > 0:
            batch_size = entity_candidate.shape[0]
            candidate_num = entity_candidate.shape[1]
            # print("negative_num=", negative_num)
            # 获取被mask实体的embedding
            batch_entity_query_embedding = list()
            for ei, input_id in enumerate(input_ids[task_id == 2]):
                batch_entity_query_embedding.append(
                    torch.mean(sequence_output[task_id == 2][ei][input_id == mask_id[task_id == 2][ei]], 0))  # [hidden_dim]
            batch_entity_query_embedding = torch.stack(batch_entity_query_embedding)  # [bz, dim]
            # print("batch_entity_query_embedding.shape=", batch_entity_query_embedding.shape)
            batch_entity_query_embedding = self.entity_mlp(batch_entity_query_embedding)  # [bz, dim]
            batch_entity_query_embedding = batch_entity_query_embedding.unsqueeze(1).repeat((1, candidate_num, 1))  # [bz, 11, dim]
            batch_entity_query_embedding = batch_entity_query_embedding.view(-1, batch_entity_query_embedding.shape[-1])  # [bz * 11, dim]
            # print("batch_entity_query_embedding.shape=", batch_entity_query_embedding.shape)

            # 获得positive和negative的BERT表示
            # entity_candidiate: [bz, 11, len]
            entity_candidate = entity_candidate.view(-1, entity_candidate.shape[-1])  # [bz * 11, len]
            entity_candidate_embedding = self.roberta.embeddings(input_ids=entity_candidate)  # [bz * 11, len, dim]
            entity_candidate_embedding = self.entity_mlp(torch.mean(entity_candidate_embedding, 1))  # [bz * 11, dim]

            contrastive_entity_label = torch.Tensor([0] * (candidate_num - 1) + [1]).float().cuda()
            contrastive_entity_label = contrastive_entity_label.unsqueeze(0).repeat([batch_size, 1]).view(-1)  # [bz * 11]

            entity_loss = self.contrastive_loss_fn(
                batch_entity_query_embedding, entity_candidate_embedding, contrastive_entity_label
            )
            total_loss.append(entity_loss)

        relation_candidate = relation_candidate[task_id == 3]
        if len(relation_candidate) > 0:
            batch_size = relation_candidate.shape[0]
            candidate_num = relation_candidate.shape[1]
            # print("negative_num=", negative_num)
            # 获取被mask relation的embedding
            batch_relation_query_embedding = list()
            for ei, input_id in enumerate(input_ids[task_id == 3]):
                batch_relation_query_embedding.append(
                    torch.mean(sequence_output[task_id == 3][ei][input_id == mask_id[task_id == 3][ei]], 0))  # [hidden_dim]
            batch_relation_query_embedding = torch.stack(batch_relation_query_embedding)  # [bz, dim]
            # print("batch_relation_query_embedding.shape=", batch_relation_query_embedding.shape)
            batch_relation_query_embedding = self.relation_mlp(batch_relation_query_embedding)  # [bz, dim]
            batch_relation_query_embedding = batch_relation_query_embedding.unsqueeze(1).repeat(
                (1, candidate_num, 1))  # [bz, 11, dim]
            batch_relation_query_embedding = batch_relation_query_embedding.view(-1, batch_relation_query_embedding.shape[-1])  # [bz * 11, dim]
            # print("batch_relation_query_embedding.shape=", batch_relation_query_embedding.shape)

            # 获得positive和negative的BERT表示
            # entity_candidiate: [bz, 11, len]
            relation_candidate = relation_candidate.view(-1, relation_candidate.shape[-1])  # [bz * 11, len]
            relation_candidate_embedding = self.roberta.embeddings(input_ids=relation_candidate)  # [bz * 11, len, dim]
            relation_candidate_embedding = self.relation_mlp(torch.mean(relation_candidate_embedding, 1))  # [bz * 11, dim]

            contrastive_relation_label = torch.Tensor([0] * (candidate_num - 1) + [1]).float().cuda()
            contrastive_relation_label = contrastive_relation_label.unsqueeze(0).repeat([batch_size, 1]).view(-1)  # [bz * 11]

            relation_loss = self.contrastive_loss_fn(
                batch_relation_query_embedding, relation_candidate_embedding, contrastive_relation_label
            )
            total_loss.append(relation_loss)

        total_loss = torch.sum(torch.stack(total_loss), -1)

        # end_time = time()
        # print("neural_mode_time: {}".format(end_time - start_time))
        # print("masked_lm_loss.unsqueeze(0)=", masked_lm_loss.unsqueeze(0))
        # print("masked_lm_loss.unsqueeze(0).shape=", masked_lm_loss.unsqueeze(0).shape)
        # print("logits=", prediction_scores.argmax(2))
        # print("logits.shape=", prediction_scores.argmax(2).shape)


        return OrderedDict([
            ("loss", total_loss),
            ("mlm_loss", masked_lm_loss.unsqueeze(0)),
            # ("noise_detect_loss", noise_detect_loss.unsqueeze(0) if noise_detect_loss is not None else None),
            # ("entity_loss", entity_loss.unsqueeze(0) if entity_loss is not None else None),
            # ("relation_loss", relation_loss.unsqueeze(0) if relation_loss is not None else None),
            ("logits", prediction_scores.argmax(2)),
            # ("noise_detect_logits", noise_detect_scores.argmax(-1) if noise_detect_scores is not None and len(noise_detect_scores) > 0 else None),
        ])


class DeBertaKPPLMForProcessedWikiKGPLM(DebertaForMaskedLM):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        # self.roberta = RobertaModel(config)
        try:
            classifier_dropout = (
                config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            )
        except:
            classifier_dropout = (config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        # self.cls = BertOnlyMLMHead(config)
        # self.lm_head = RobertaLMHead(config)  # Masked Language Modeling head
        self.detector = NSPHead(config)  # Knowledge Noise Detection head
        self.entity_mlp = nn.Linear(config.hidden_size, config.hidden_size)
        self.relation_mlp = nn.Linear(config.hidden_size, config.hidden_size)
        # self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, config.num_ner_labels) for _ in range(config.entity_type_num)])

        self.contrastive_loss_fn = ContrastiveLoss()
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            # entity_label=None,
            entity_candidate=None,
            # relation_label=None,
            relation_candidate=None,
            noise_detect_label=None,
            task_id=None,
            mask_id=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # start_time = time()
        mlm_labels = labels

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print("attention_mask.shape=", attention_mask.shape)
        # print("input_ids[0]=", input_ids[0])
        # print("token_type_ids[0]=", token_type_ids[0])
        # attention_mask = None

        outputs = self.deberta(
            input_ids,
            # attention_mask=attention_mask,
            attention_mask=None,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)  # mlm head
        # noise_detect_scores = self.detector(pooled_output)  # knowledge noise detector use pool output
        noise_detect_scores = self.detector(sequence_output[:, 0, :])  # knowledge noise detector use cls embedding

        # ner
        # sequence_output = self.dropout(sequence_output)
        # ner_logits = torch.stack([classifier(sequence_output) for classifier in self.classifiers]).movedim(1, 0)

        # mlm
        masked_lm_loss, noise_detect_loss, entity_loss, total_loss = None, None, None, None
        total_loss = list()
        if mlm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            total_loss.append(masked_lm_loss)

        # if noise_detect_label is not None:
        #     noise_detect_scores = noise_detect_scores[task_id == 1]
        #     noise_detect_label = noise_detect_label[task_id == 1]
        #
        #     if len(noise_detect_label) > 0:
        #         loss_fct = CrossEntropyLoss()
        #         noise_detect_loss = loss_fct(noise_detect_scores.view(-1, 2), noise_detect_label.view(-1))
        #         total_loss.append(noise_detect_loss)

        entity_candidate = entity_candidate[task_id == 2]
        if len(entity_candidate) > 0:
            batch_size = entity_candidate.shape[0]
            candidate_num = entity_candidate.shape[1]
            # print("negative_num=", negative_num)
            # 获取被mask实体的embedding
            batch_entity_query_embedding = list()
            for ei, input_id in enumerate(input_ids[task_id == 2]):
                batch_entity_query_embedding.append(
                    torch.mean(sequence_output[task_id == 2][ei][input_id == mask_id[task_id == 2][ei]], 0))  # [hidden_dim]
            batch_entity_query_embedding = torch.stack(batch_entity_query_embedding)  # [bz, dim]
            # print("batch_entity_query_embedding.shape=", batch_entity_query_embedding.shape)
            batch_entity_query_embedding = self.entity_mlp(batch_entity_query_embedding)  # [bz, dim]
            batch_entity_query_embedding = batch_entity_query_embedding.unsqueeze(1).repeat((1, candidate_num, 1))  # [bz, 11, dim]
            batch_entity_query_embedding = batch_entity_query_embedding.view(-1, batch_entity_query_embedding.shape[-1])  # [bz * 11, dim]
            # print("batch_entity_query_embedding.shape=", batch_entity_query_embedding.shape)

            # 获得positive和negative的BERT表示
            # entity_candidiate: [bz, 11, len]
            entity_candidate = entity_candidate.view(-1, entity_candidate.shape[-1])  # [bz * 11, len]
            entity_candidate_embedding = self.deberta.embeddings(input_ids=entity_candidate)  # [bz * 11, len, dim]
            entity_candidate_embedding = self.entity_mlp(torch.mean(entity_candidate_embedding, 1))  # [bz * 11, dim]

            contrastive_entity_label = torch.Tensor([0] * (candidate_num - 1) + [1]).float().cuda()
            contrastive_entity_label = contrastive_entity_label.unsqueeze(0).repeat([batch_size, 1]).view(-1)  # [bz * 11]

            entity_loss = self.contrastive_loss_fn(
                batch_entity_query_embedding, entity_candidate_embedding, contrastive_entity_label
            )
            total_loss.append(entity_loss)

        relation_candidate = relation_candidate[task_id == 3]
        if len(relation_candidate) > 0:
            batch_size = relation_candidate.shape[0]
            candidate_num = relation_candidate.shape[1]
            # print("negative_num=", negative_num)
            # 获取被mask relation的embedding
            batch_relation_query_embedding = list()
            for ei, input_id in enumerate(input_ids[task_id == 3]):
                batch_relation_query_embedding.append(
                    torch.mean(sequence_output[task_id == 3][ei][input_id == mask_id[task_id == 3][ei]], 0))  # [hidden_dim]
            batch_relation_query_embedding = torch.stack(batch_relation_query_embedding)  # [bz, dim]
            # print("batch_relation_query_embedding.shape=", batch_relation_query_embedding.shape)
            batch_relation_query_embedding = self.relation_mlp(batch_relation_query_embedding)  # [bz, dim]
            batch_relation_query_embedding = batch_relation_query_embedding.unsqueeze(1).repeat(
                (1, candidate_num, 1))  # [bz, 11, dim]
            batch_relation_query_embedding = batch_relation_query_embedding.view(-1, batch_relation_query_embedding.shape[-1])  # [bz * 11, dim]
            # print("batch_relation_query_embedding.shape=", batch_relation_query_embedding.shape)

            # 获得positive和negative的BERT表示
            # entity_candidiate: [bz, 11, len]
            relation_candidate = relation_candidate.view(-1, relation_candidate.shape[-1])  # [bz * 11, len]
            relation_candidate_embedding = self.deberta.embeddings(input_ids=relation_candidate)  # [bz * 11, len, dim]
            relation_candidate_embedding = self.relation_mlp(torch.mean(relation_candidate_embedding, 1))  # [bz * 11, dim]

            contrastive_relation_label = torch.Tensor([0] * (candidate_num - 1) + [1]).float().cuda()
            contrastive_relation_label = contrastive_relation_label.unsqueeze(0).repeat([batch_size, 1]).view(-1)  # [bz * 11]

            relation_loss = self.contrastive_loss_fn(
                batch_relation_query_embedding, relation_candidate_embedding, contrastive_relation_label
            )
            total_loss.append(relation_loss)

        total_loss = torch.sum(torch.stack(total_loss), -1)

        # end_time = time()
        # print("neural_mode_time: {}".format(end_time - start_time))
        # print("masked_lm_loss.unsqueeze(0)=", masked_lm_loss.unsqueeze(0))
        # print("masked_lm_loss.unsqueeze(0).shape=", masked_lm_loss.unsqueeze(0).shape)
        # print("logits=", prediction_scores.argmax(2))
        # print("logits.shape=", prediction_scores.argmax(2).shape)


        return OrderedDict([
            ("loss", total_loss),
            ("mlm_loss", masked_lm_loss.unsqueeze(0)),
            # ("noise_detect_loss", noise_detect_loss.unsqueeze(0) if noise_detect_loss is not None else None),
            # ("entity_loss", entity_loss.unsqueeze(0) if entity_loss is not None else None),
            # ("relation_loss", relation_loss.unsqueeze(0) if relation_loss is not None else None),
            ("logits", prediction_scores.argmax(2)),
            # ("noise_detect_logits", noise_detect_scores.argmax(-1) if noise_detect_scores is not None and len(noise_detect_scores) > 0 else None),
        ])


class RoBertaForWikiKGPLM(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # self.cls = BertOnlyMLMHead(config)
        self.lm_head = RobertaLMHead(config) # Masked Language Modeling head
        self.detector = NSPHead(config) # Knowledge Noise Detection head
        self.entity_mlp = nn.Linear(config.hidden_size, config.hidden_size)
        self.relation_mlp = nn.Linear(config.hidden_size, config.hidden_size)
        # self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, config.num_ner_labels) for _ in range(config.entity_type_num)])

        self.contrastive_loss_fn = ContrastiveLoss()
        self.post_init()

        self.tokenizer = RobertaTokenizer.from_pretrained(config.name_or_path)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            mlm_labels=None,
            entity_label=None,
            entity_negative=None,
            relation_label=None,
            relation_negative=None,
            noise_detect_label=None,
            task_id=None,
            mask_id=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # start_time = time()

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print("attention_mask.shape=", attention_mask.shape)
        # print("input_ids[0]=", input_ids[0])
        # print("token_type_ids[0]=", token_type_ids[0])
        # attention_mask = None


        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.lm_head(sequence_output) # mlm head
        noise_detect_scores = self.detector(pooled_output) # knowledge noise detector


        # ner
        # sequence_output = self.dropout(sequence_output)
        # ner_logits = torch.stack([classifier(sequence_output) for classifier in self.classifiers]).movedim(1, 0)

        # mlm
        masked_lm_loss, noise_detect_loss, entity_loss, total_loss = None, None, None, None
        if mlm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))

        if noise_detect_label is not None:
            loss_fct = CrossEntropyLoss()
            noise_detect_loss = loss_fct(noise_detect_scores.view(-1, 2), noise_detect_label.view(-1))
            total_loss = masked_lm_loss + noise_detect_loss

        if entity_label is not None and entity_negative is not None:
            batch_size = input_ids.shape[0]
            negative_num = entity_negative.shape[1]
            # print("negative_num=", negative_num)
            # 获取被mask实体的embedding
            batch_query_embedding = list()
            for ei, input_id in enumerate(input_ids):
                batch_query_embedding.append(torch.mean(sequence_output[ei][input_id == mask_id[ei]], 0)) # [hidden_dim]
            batch_query_embedding = torch.stack(batch_query_embedding) # [bz, dim]
            # print("batch_query_embedding.shape=", batch_query_embedding.shape)
            batch_query_embedding = self.entity_mlp(batch_query_embedding) # [bz, dim]
            batch_query_embedding = batch_query_embedding.unsqueeze(1).repeat((1, negative_num + 1, 1)) # [bz, 11, dim]
            batch_query_embedding = batch_query_embedding.view(-1, batch_query_embedding.shape[-1]) # [bz * 11, dim]
            # print("batch_query_embedding.shape=", batch_query_embedding.shape)

            # 获得positive和negative的BERT表示
            # entity_label: [bz, len], entity_negative: [bz, 10, len]
            entity_negative = entity_negative.view(-1, entity_negative.shape[-1]) # [bz * 10, len]
            entity_label_embedding = self.roberta.embeddings(input_ids=entity_label) # [bz, len, dim]
            entity_label_embedding = self.entity_mlp(torch.mean(entity_label_embedding, 1)) # [bz, dim]
            entity_label_embedding = entity_label_embedding.unsqueeze(1) # [bz, 1, dim]

            entity_negative_embedding = self.roberta.embeddings(input_ids=entity_negative) # [bz * 10, len, dim]
            entity_negative_embedding = self.entity_mlp(torch.mean(entity_negative_embedding, 1)) # [bz * 10, dim]
            entity_negative_embedding = entity_negative_embedding \
                .view(input_ids.shape[0], -1, entity_negative_embedding.shape[-1]) # [bz, 10, dim]

            contrastive_label = torch.Tensor([0] * negative_num + [1]).float().cuda()
            contrastive_label = contrastive_label.unsqueeze(0).repeat([batch_size, 1]).view(-1) # [bz * 11]
            # print("entity_negative_embedding.shape=", entity_negative_embedding.shape)
            # print("entity_label_embedding.shape=", entity_label_embedding.shape)
            candidate_embedding = torch.cat([entity_negative_embedding, entity_label_embedding], 1) # [bz, 11, dim]
            candidate_embedding = candidate_embedding.view(-1, candidate_embedding.shape[-1]) # [bz * 11, dim]
            # print("candidate_embedding.shape=", candidate_embedding.shape)

            entity_loss = self.contrastive_loss_fn(batch_query_embedding, candidate_embedding, contrastive_label)
            total_loss = masked_lm_loss + entity_loss


        # if ner_labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     # Only keep active parts of the loss
        #
        #     active_loss = attention_mask.repeat(self.config.entity_type_num, 1, 1).view(-1) == 1
        #     active_logits = ner_logits.reshape(-1, self.config.num_ner_labels)
        #     active_labels = torch.where(
        #         active_loss, ner_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(ner_labels)
        #     )
        #     ner_loss = loss_fct(active_logits, active_labels)
        #
        # if masked_lm_loss:
        #     total_loss = masked_lm_loss + ner_loss * 4
        # print("total_loss=", total_loss)
        # print("mlm_loss=", masked_lm_loss)


        # end_time = time()
        # print("neural_mode_time: {}".format(end_time - start_time))

        return OrderedDict([
            ("loss", total_loss),
            ("mlm_loss", masked_lm_loss.unsqueeze(0)),
            ("noise_detect_loss", noise_detect_loss.unsqueeze(0) if noise_detect_loss is not None else None),
            ("entity_loss", entity_loss.unsqueeze(0) if entity_label is not None else None),
            ("logits", prediction_scores.argmax(2)),
            ("noise_detect_logits", noise_detect_scores.argmax(-1) if noise_detect_scores is not None else None),
        ])
        # MaskedLMOutput(
        #     loss=total_loss,
        #     logits=prediction_scores.argmax(2),
        #     ner_l
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )




class BertForWikiKGPLM(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # self.cls = BertOnlyMLMHead(config)
        self.cls = BertPreTrainedModel(config)
        self.entity_mlp = nn.Linear(config.hidden_size, config.hidden_size)
        self.relation_mlp = nn.Linear(config.hidden_size, config.hidden_size)
        # self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, config.num_ner_labels) for _ in range(config.entity_type_num)])

        self.contrastive_loss_fn = ContrastiveLoss()
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            mlm_labels=None,
            entity_label=None,
            entity_negative=None,
            relation_label=None,
            relation_negative=None,
            noise_detect_label=None,
            task_id=None,
            mask_id=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        print("attention_mask.shape=", attention_mask.shape)
        print("input_ids[0]=", input_ids[0])
        print("token_type_ids[0]=", token_type_ids[0])
        attention_mask = None
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        # ner
        # sequence_output = self.dropout(sequence_output)
        # ner_logits = torch.stack([classifier(sequence_output) for classifier in self.classifiers]).movedim(1, 0)

        # mlm
        masked_lm_loss, noise_detect_loss, entity_loss, total_loss = None, None, None, None

        if mlm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))

        if noise_detect_label is not None:
            loss_fct = CrossEntropyLoss()
            noise_detect_loss = loss_fct(seq_relationship_score.view(-1, 2), noise_detect_label.view(-1))
            total_loss = masked_lm_loss + noise_detect_loss

        if entity_label is not None and entity_negative is not None:
            negative_num = entity_negative.shape[1]
            # 获取被mask实体的embedding
            batch_query_embedding = list()
            for ei, input_id in enumerate(input_ids):
                batch_query_embedding.append(torch.mean(sequence_output[ei][input_id == mask_id[ei]], 0)) # [hidden_dim]
            batch_query_embedding = torch.stack(batch_query_embedding) # [bz, dim]
            batch_query_embedding = self.entity_mlp(batch_query_embedding) # [bz, dim]
            batch_query_embedding = batch_query_embedding.repeat((1, negative_num + 1, 1)) # [bz, 11, dim]

            # 获得positive和negative的BERT表示
            # entity_label: [bz, len], entity_negative: [bz, 10, len]
            entity_negative = entity_negative.view(-1, entity_negative.shape[-1]) # [bz * 10, len]
            entity_label_embedding = self.bert.embeddings(input_id=entity_label) # [bz, len, dim]
            entity_label_embedding = self.entity_mlp(torch.mean(entity_label_embedding, 1)) # [bz, dim]
            entity_label_embedding = entity_label_embedding.unsqueeze(1) # [bz, 1, dim]

            entity_negative_embedding = self.bert.embeddings(input_id=entity_negative) # [bz * 10, len, dim]
            entity_negative_embedding = self.entity_mlp(torch.mean(entity_negative_embedding, 1)) # [bz * 10, dim]
            entity_negative_embedding = entity_negative_embedding \
                .view(input_ids.shape[0], -1, entity_negative_embedding.shape[-1]) # [bz, 10, dim]

            contrastive_label = torch.Tensor([0] * negative_num + [1]).float().cuda()
            candidate_embedding = torch.cat([entity_negative_embedding, entity_label_embedding], 1) # [bz, 11, dim]

            entity_loss = self.contrastive_loss_fn(batch_query_embedding, candidate_embedding, contrastive_label)
            total_loss = masked_lm_loss + entity_loss


        # if ner_labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     # Only keep active parts of the loss
        #
        #     active_loss = attention_mask.repeat(self.config.entity_type_num, 1, 1).view(-1) == 1
        #     active_logits = ner_logits.reshape(-1, self.config.num_ner_labels)
        #     active_labels = torch.where(
        #         active_loss, ner_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(ner_labels)
        #     )
        #     ner_loss = loss_fct(active_logits, active_labels)
        #
        # if masked_lm_loss:
        #     total_loss = masked_lm_loss + ner_loss * 4

        return OrderedDict([
            ("loss", total_loss),
            ("mlm_loss", masked_lm_loss.unsqueeze(0)),
            ("noise_detect_loss", noise_detect_loss.unsqueeze(0)),
            ("entity_loss", entity_loss.unsqueeze(0)),
            ("logits", prediction_scores.argmax(2)),
            ("noise_detect_logits", seq_relationship_score.argmax(3)),
            ()
        ])
        # MaskedLMOutput(
        #     loss=total_loss,
        #     logits=prediction_scores.argmax(2),
        #     ner_l
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )
