import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel
from transformers.models.albert.modeling_albert import AlbertPreTrainedModel, AlbertModel
from transformers.models.megatron_bert.modeling_megatron_bert import MegatronBertPreTrainedModel, MegatronBertModel
from torch.nn import CrossEntropyLoss
from loss.focal_loss import FocalLoss
from loss.label_smoothing import LabelSmoothingCrossEntropy
from models.basic_modules.crf import CRF
from tools.model_utils.parameter_freeze import ParameterFreeze

from tools.runner_utils.log_util import logging
logger = logging.getLogger(__name__)

freezer = ParameterFreeze()


"""
BERT for token-level classification with softmax head.
"""
class BertSoftmaxForSequenceLabeling(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmaxForSequenceLabeling, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        if self.config.use_freezing:
            self.bert = freezer.freeze_lm(self.bert)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ["lsr", "focal", "ce"]
            if self.loss_type == "lsr":
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == "focal":
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


"""
RoBERTa for token-level classification with softmax head.
"""
class RobertaSoftmaxForSequenceLabeling(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaSoftmaxForSequenceLabeling, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        if self.config.use_freezing:
            self.roberta = freezer.freeze_lm(self.roberta)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.roberta(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ["lsr", "focal", "ce"]
            if self.loss_type == "lsr":
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == "focal":
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


"""
ALBERT for token-level classification with softmax head.
"""
class AlbertSoftmaxForSequenceLabeling(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertSoftmaxForSequenceLabeling, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.bert = AlbertModel(config)
        if self.config.use_freezing:
            self.bert = freezer.freeze_lm(self.bert)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,
                            position_ids=position_ids,head_mask=head_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ["lsr", "focal", "ce"]
            if self.loss_type =="lsr":
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == "focal":
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


"""
MegatronBERT for token-level classification with softmax head.
"""
class MegatronBertSoftmaxForSequenceLabeling(MegatronBertPreTrainedModel):
    def __init__(self, config):
        super(MegatronBertSoftmaxForSequenceLabeling, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = MegatronBertModel(config)
        if self.config.use_freezing:
            self.bert = freezer.freeze_lm(self.bert)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ["lsr", "focal", "ce"]
            if self.loss_type == "lsr":
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == "focal":
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


"""
BERT for token-level classification with CRF head.
"""
class BertCrfForSequenceLabeling(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForSequenceLabeling, self).__init__(config)
        self.bert = BertModel(config)
        if self.config.use_freezing:
            self.bert = freezer.freeze_lm(self.bert)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None,input_lens=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores


"""
RoBERTa for token-level classification with CRF head.
"""
class RobertaCrfForSequenceLabeling(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaCrfForSequenceLabeling, self).__init__(config)
        self.roberta = RobertaModel(config)
        if self.config.use_freezing:
            self.roberta = freezer.freeze_lm(self.roberta)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None,input_lens=None):
        outputs =self.roberta(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores


"""
ALBERT for token-level classification with CRF head.
"""
class AlbertCrfForSequenceLabeling(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertCrfForSequenceLabeling, self).__init__(config)
        self.bert = AlbertModel(config)
        if self.config.use_freezing:
            self.bert = freezer.freeze_lm(self.bert)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None,input_lens=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores


"""
MegatronBERT for token-level classification with CRF head.
"""
class MegatronBertCrfForSequenceLabeling(MegatronBertPreTrainedModel):
    def __init__(self, config):
        super(MegatronBertCrfForSequenceLabeling, self).__init__(config)
        self.bert = MegatronBertModel(config)
        if self.config.use_freezing:
            self.bert = freezer.freeze_lm(self.bert)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None,input_lens=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores
