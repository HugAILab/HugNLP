"""Custom models for few-shot learning specific operations."""

import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertForSequenceClassification, BertModel, BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaLMHead, RobertaClassificationHead, RobertaPreTrainedModel
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model, StableDropout, ContextPooler, DebertaV2OnlyMLMHead
from transformers.models.deberta.modeling_deberta import DebertaPreTrainedModel, DebertaModel, StableDropout, ContextPooler, DebertaOnlyMLMHead
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.configuration_bert import BertConfig
import logging
from models.basic_modules.adapter import RobertaAdaModel, BertAdaModel
import os
from models.basic_modules.prefix_encoder import PrefixEncoder
from tools.model_utils.parameter_freeze import ParameterFreeze

freezer = ParameterFreeze()

logger = logging.getLogger(__name__)

# Note: 如果mask_pos为None，请检查输入的模板是否有<mask>标记，是否修改data_collator文件

"""
Vanilla Prompt-tuning BERT
"""
class PromptBertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pre_seq_len = self.config.pre_seq_len
        self.hidden_size = self.config.hidden_size
        # backbone
        self.bert = BertModel(config)
        if self.config.use_freezing:
            self.bert = freezer.freeze_lm(self.bert)
        # mlm head
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.label_word_list = torch.Tensor(self.config.label_word_list).long().to(self.bert.device)

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def freeze_backbone(self, use_freezing: bool=True):
        if use_freezing:
            self.bert = freezer.freeze_lm(self.bert)
        else:
            self.bert = freezer.unfreeze_lm(self.bert)

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        """
        Encoding and obtain logits at masked position
        """
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
        # Encode everything
        if inputs_embeds is None:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.bert(
                None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds
            )
        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        return logits, sequence_mask_output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        block_flag=None,
        return_dict=None,
    ):

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction="batchmean")
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        if not return_dict:
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )



"""
P-tuning BERT
"""
class PromptBertPtuningForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pre_seq_len = self.config.pre_seq_len
        self.hidden_size = self.config.hidden_size
        # backbone
        self.bert = BertModel(config)
        if self.config.use_freezing:
            self.bert = freezer.freeze_lm(self.bert)
        # mlm head
        self.cls = BertOnlyMLMHead(config)
        # prompt encoder
        self.prompt_encoder = None
        # plm embedding layer
        self.backbone_embeddings = self.bert.embeddings.word_embeddings
        # prompt embedding layer
        self.prompt_embeddings = torch.nn.Embedding(self.pre_seq_len, self.hidden_size)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.label_word_list = torch.Tensor(self.config.label_word_list).long().to(self.bert.device)

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def freeze_backbone(self, use_freezing: bool=True):
        if use_freezing:
            self.bert = freezer.freeze_lm(self.bert)
        else:
            self.bert = freezer.unfreeze_lm(self.bert)


    def generate_continuous_prompt_inputs(self, input_ids, block_flag=None, reparameterization=False):
        """
        Generate continuous prompt embedding
        """
        inputs_embeds = self.backbone_embeddings(input_ids)

        batch_size = inputs_embeds.shape[0]
        if block_flag is None:
            # the first token is set 1, others are set 0
            block_flag = torch.zeros_like(input_ids).long().to(inputs_embeds.device)
            block_flag[:, 0] = 1
        try:
            replace_embeds = self.prompt_embeddings(
                torch.LongTensor(list(range(self.pre_seq_len))).to(inputs_embeds.device))
        except:
            import pdb
            pdb.set_trace()
            replace_embeds = self.prompt_embeddings(
                torch.LongTensor(list(range(self.pre_seq_len))))
        replace_embeds = replace_embeds.unsqueeze(0)  # [batch_size, prompt_length, embed_size]

        if self.prompt_encoder is not None:
            replace_embeds = self.prompt_encoder(replace_embeds)

        # edit by wjn
        if reparameterization:
            # blocked_indices = (block_flag == 1).nonzero(as_tuple=False).reshape((batch_size, self.pre_seq_len, 2))[:, :, 1]
            blocked_indices = (block_flag == 1).nonzero()
            # reparameterization
            for bidx in range(batch_size):
                for i in range(blocked_indices.shape[1]):
                    inputs_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[:, i, :].squeeze()
        else:
            replace_embeds = replace_embeds.expand(batch_size, self.pre_seq_len, -1).to(inputs_embeds.device)
            inputs_embeds = torch.cat((replace_embeds, inputs_embeds), dim=1)
        return inputs_embeds

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        """
        Encoding and obtain logits at masked position
        """
        batch_size = inputs_embeds.shape[0]
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
        # Encode everything
        if inputs_embeds is None:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:

            if inputs_embeds.shape[1] == attention_mask.shape[1]:
                outputs = self.bert(
                    None,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    inputs_embeds=inputs_embeds
                )
                # Get <mask> token representation
                sequence_output, pooled_output = outputs[:2]
                # sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
            else:
                if attention_mask is not None:
                    prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).long().to(self.bert.device)
                    attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
                if token_type_ids is not None:
                    prefix_token_type_ids = torch.zeros(batch_size, self.pre_seq_len).long().to(self.bert.device)
                    token_type_ids = torch.cat((prefix_token_type_ids, token_type_ids), dim=1)
                outputs = self.bert(
                    None,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    inputs_embeds=inputs_embeds
                )
                # Get <mask> token representation
                sequence_output, pooled_output = outputs[:2]
                sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()

        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        return logits, sequence_mask_output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        block_flag=None,
        return_dict=None,
    ):

        inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)
        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction="batchmean")
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        if not return_dict:
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )



"""
Prefix-tuning BERT
"""
class PromptBertPrefixForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)


        self.num_labels = config.num_labels
        self.pre_seq_len = self.config.pre_seq_len
        self.hidden_size = self.config.hidden_size

        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        # backbone
        self.bert = BertModel(config)
        if self.config.use_freezing:
            self.bert = freezer.freeze_lm(self.bert)
        # mlm head
        self.cls = BertOnlyMLMHead(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # plm embedding layer
        self.backbone_embeddings = self.bert.embeddings.word_embeddings
        # prompt embedding layer
        self.prompt_embeddings = torch.nn.Embedding(self.pre_seq_len, self.hidden_size)
        # prefix encoder
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.label_word_list = torch.Tensor(self.config.label_word_list).long().to(self.bert.device)

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def freeze_backbone(self, use_freezing: bool=True):
        if use_freezing:
            self.bert = freezer.freeze_lm(self.bert)
        else:
            self.bert = freezer.unfreeze_lm(self.bert)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def embed_encode(self, input_ids):
        embedding_output = self.bert.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        batch_size = input_ids.size(0)

        # add prefix for prompt-tuning
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
        )
        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        # sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # print("prediction_mask_scores.shape=", prediction_mask_scores.shape) # [batch_size, seq_len, vocab_size]

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        return logits, sequence_mask_output


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        block_flag=None,
        return_dict=None,
    ):

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction="batchmean")
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        if not return_dict:
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


"""
Adapter-tuning BERT
"""
class PromptBertAdapterForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertAdaModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        if self.config.use_freezing:
            self.bert = freezer.freeze_lm_component(self.bert, "adapter")

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = torch.Tensor(self.config.label_word_list).long().to(self.bert.device)

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def freeze_backbone(self, use_freezing: bool=True):
        if use_freezing:
            self.bert = freezer.freeze_lm_component(self.bert, "adapter")
        else:
            self.bert = freezer.unfreeze_lm(self.bert)

    def embed_encode(self, input_ids):
        embedding_output = self.bert.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        if inputs_embeds is None:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.bert(
                None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds
            )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        return logits, sequence_mask_output


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        block_flag=None,
        return_dict=None,
    ):

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction="batchmean")
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        if not return_dict:
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )



"""
Vanilla Prompt-tuning RoBERTa
"""
class PromptRobertaForSequenceClassification(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pre_seq_len = self.config.pre_seq_len
        self.hidden_size = self.config.hidden_size
        # backbone
        self.roberta = RobertaModel(config)
        if self.config.use_freezing:
            self.roberta = freezer.freeze_lm(self.roberta)
        # mlm head
        self.cls = RobertaLMHead(config)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.label_word_list = torch.Tensor(self.config.label_word_list).long().to(self.roberta.device)

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def freeze_backbone(self, use_freezing: bool=True):
        if use_freezing:
            self.roberta = freezer.freeze_lm(self.roberta)
        else:
            self.roberta = freezer.unfreeze_lm(self.roberta)

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        """
        Encoding and obtain logits at masked position
        """
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
        # Encode everything
        if inputs_embeds is None:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.roberta(
                None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds
            )
        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        return logits, sequence_mask_output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        block_flag=None,
        return_dict=None,
    ):

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction="batchmean")
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        if not return_dict:
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


"""
P-tuning RoBERTa
"""
class PromptRobertaPtuningForSequenceClassification(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pre_seq_len = self.config.pre_seq_len
        self.hidden_size = self.config.hidden_size
        # backbone
        self.roberta = RobertaModel(config)
        if self.config.use_freezing:
            self.roberta = freezer.freeze_lm(self.roberta)
        # mlm head
        self.cls = RobertaLMHead(config)
        # prompt encoder
        self.prompt_encoder = None
        # plm embedding layer
        self.backbone_embeddings = self.roberta.embeddings.word_embeddings
        # prompt embedding layer
        self.prompt_embeddings = torch.nn.Embedding(self.pre_seq_len, self.hidden_size)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.label_word_list = torch.Tensor(self.config.label_word_list).long().to(self.roberta.device)

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def freeze_backbone(self, use_freezing: bool=True):
        if use_freezing:
            self.roberta = freezer.freeze_lm(self.roberta)
        else:
            self.roberta = freezer.unfreeze_lm(self.roberta)


    def generate_continuous_prompt_inputs(self, input_ids, block_flag=None, reparameterization=False):
        """
        Generate continuous prompt embedding
        """
        inputs_embeds = self.backbone_embeddings(input_ids)

        batch_size = inputs_embeds.shape[0]
        if block_flag is None:
            # the first token is set 1, others are set 0
            block_flag = torch.zeros_like(input_ids).long().to(inputs_embeds.device)
            block_flag[:, 0] = 1
        try:
            replace_embeds = self.prompt_embeddings(
                torch.LongTensor(list(range(self.pre_seq_len))).to(inputs_embeds.device))
        except:
            import pdb
            pdb.set_trace()
            replace_embeds = self.prompt_embeddings(torch.LongTensor(list(range(self.pre_seq_len))))
        replace_embeds = replace_embeds.unsqueeze(0)  # [batch_size, prompt_length, embed_size]

        if self.prompt_encoder is not None:
            replace_embeds = self.prompt_encoder(replace_embeds)

        # edit by wjn
        if reparameterization:
            # blocked_indices = (block_flag == 1).nonzero(as_tuple=False).reshape((batch_size, self.pre_seq_len, 2))[:, :, 1]
            blocked_indices = (block_flag == 1).nonzero()
            # reparameterization
            for bidx in range(batch_size):
                for i in range(blocked_indices.shape[1]):
                    inputs_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[:, i, :].squeeze()
        else:
            replace_embeds = replace_embeds.expand(batch_size, self.pre_seq_len, -1).to(inputs_embeds.device)
            inputs_embeds = torch.cat((replace_embeds, inputs_embeds), dim=1)
        return inputs_embeds

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        """
        Encoding and obtain logits at masked position
        """
        batch_size = inputs_embeds.shape[0]
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
        # Encode everything
        if inputs_embeds is None:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:

            if inputs_embeds.shape[1] == attention_mask.shape[1]:
                outputs = self.roberta(
                    None,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    inputs_embeds=inputs_embeds
                )
                # Get <mask> token representation
                sequence_output, pooled_output = outputs[:2]
                # sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
            else:
                if attention_mask is not None:
                    prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).long().to(self.roberta.device)
                    attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
                if token_type_ids is not None:
                    prefix_token_type_ids = torch.zeros(batch_size, self.pre_seq_len).long().to(self.roberta.device)
                    token_type_ids = torch.cat((prefix_token_type_ids, token_type_ids), dim=1)
                outputs = self.roberta(
                    None,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    inputs_embeds=inputs_embeds
                )
                # Get <mask> token representation
                sequence_output, pooled_output = outputs[:2]
                sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()

        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]
        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        return logits, sequence_mask_output

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        block_flag=None,
        return_dict=None,
    ):

        inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)
        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction="batchmean")
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        if not return_dict:
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


"""
Prefix-tuning RoBERTa
"""
class PromptRobertaPrefixForSequenceClassification(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)


        self.num_labels = config.num_labels
        self.pre_seq_len = self.config.pre_seq_len
        self.hidden_size = self.config.hidden_size

        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        # backbone
        self.robert = RobertaModel(config)
        if self.config.use_freezing:
            self.robert = freezer.freeze_lm(self.robert)
        # mlm head
        self.cls = RobertaLMHead(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # plm embedding layer
        self.backbone_embeddings = self.robert.embeddings.word_embeddings
        # prompt embedding layer
        self.prompt_embeddings = torch.nn.Embedding(self.pre_seq_len, self.hidden_size)
        # prefix encoder
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.label_word_list = torch.Tensor(self.config.label_word_list).long().to(self.robert.device)

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def freeze_backbone(self, use_freezing: bool=True):
        if use_freezing:
            self.robert = freezer.freeze_lm(self.robert)
        else:
            self.robert = freezer.unfreeze_lm(self.robert)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.robert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def embed_encode(self, input_ids):
        embedding_output = self.robert.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        batch_size = input_ids.size(0)

        # add prefix for prompt-tuning
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.robert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.robert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
        )
        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        # sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        return logits, sequence_mask_output


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        block_flag=None,
        return_dict=None,
    ):

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction="batchmean")
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        if not return_dict:
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

"""
Adapter-tuning RoBERTa
"""
class PromptRobertaAdapterForSequenceClassification(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaAdaModel(config)
        self.cls = RobertaLMHead(config)
        self.init_weights()

        if self.config.use_freezing:
            self.roberta = freezer.freeze_lm_component(self.roberta, "adapter")

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = torch.Tensor(self.config.label_word_list).long().to(self.roberta.device)

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None

    def freeze_backbone(self, use_freezing: bool=True):
        if use_freezing:
            self.roberta = freezer.freeze_lm_component(self.roberta, "adapter")
        else:
            self.roberta = freezer.unfreeze_lm(self.berobertart)

    def embed_encode(self, input_ids):
        embedding_output = self.roberta.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        if inputs_embeds is None:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.roberta(
                None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds
            )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        return logits, sequence_mask_output


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        block_flag=None,
        return_dict=None,
    ):

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction="batchmean")
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        if not return_dict:
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


# class DebertaForPromptFinetuning(DebertaPreTrainedModel):
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#     _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         #self.deberta = DebertaV2Model(config)

#         self.deberta = DebertaModel(config)
#         self.cls = DebertaOnlyMLMHead(config)

#         if self.config.use_freezing:
#             self.deberta = freezer.freeze_lm(self.deberta)

#         self.pooler = ContextPooler(config)
#         output_dim = self.pooler.output_dim

#         self.classifier = torch.nn.Linear(output_dim, self.num_labels)
#         drop_out = getattr(config, "cls_dropout", None)
#         drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

#         self.dropout = StableDropout(drop_out)

#         classification_list = [self.pooler, self.dropout,self.classifier]

#         self.classifier = nn.Sequential(*classification_list)
#         # self.cls = DebertaV2OnlyMLMHead(config)

#         self.map = nn.Linear(config.hidden_size, config.hidden_size)
#         self.init_weights()

#         # These attributes should be assigned once the model is initialized
#         self.model_args = None
#         self.data_args = None
#         self.label_word_list = torch.Tensor(self.config.label_word_list).long().to(self.bert.device)
#         self.K = 1
#         self.step_size=1e-5
#         # import pdb
#         # pdb.set_trace()
#         #self.step_size=config.step_size

#         # For regression
#         self.lb = None
#         self.ub = None

#         self.pre_seq_len = self.config.pre_seq_len
#         # For auto label search.
#         self.return_full_softmax = None

#     def freeze_backbone(self, use_freezing: bool=True):
#         if use_freezing:
#             self.deberta = freezer.freeze_lm(self.deberta)
#         else:
#             self.deberta = freezer.unfreeze_lm(self.deberta)



#     def embed_encode(self, input_ids):
#         embedding_output = self.deberta.embeddings.word_embeddings(input_ids)
#         return embedding_output

#     def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None,
#                return_full_softmax=False):
#         batch_size = input_ids.size(0)

#         if mask_pos is not None:
#             mask_pos = mask_pos.squeeze()


#         # Encode everything
#         if inputs_embeds is None:
#             outputs = self.deberta(
#                 input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids
#             )
#         else:
#             outputs = self.deberta(
#                 None,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#                 inputs_embeds=inputs_embeds
#             )

#         # Get <mask> token representation
#         sequence_output = outputs[0]
#         sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
#         sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

#         # Logits over vocabulary tokens
#         prediction_mask_scores = self.cls(sequence_mask_output)

#         # sequence_mask_output = self.lm_head.dense(sequence_mask_output)

#         # Exit early and only return mask logits.
#         if return_full_softmax:
#             return prediction_mask_scores

#         # Return logits for each label
#         logits = []
#         for label_id in range(len(self.label_word_list)):
#             logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
#         logits = torch.cat(logits, -1)

#         # Regression task
#         if self.config.num_labels == 1:
#             logsoftmax = nn.LogSoftmax(-1)
#             logits = logsoftmax(logits)  # Log prob of right polarity

#         if self.model_args.hybrid == 1:
#             cls_logits = self.classifier(sequence_output)
#             return (logits, cls_logits), sequence_mask_output

#         return logits, sequence_mask_output

#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             mask_pos=None,
#             labels=None,
#             inputs_embeds=None,
#             fwd_type=0,
#             block_flag=None
#     ):

#         if fwd_type == 2:
#             assert inputs_embeds is not None
#             return self.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
#                                mask_pos=mask_pos, inputs_embeds=inputs_embeds)

#         elif fwd_type == 1:
#             return self.embed_encode(input_ids)



#         if (self.model_args.prompt_ptuning or self.model_args.prompt_prefix) and block_flag is not None:
#             inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)

#         logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)

#         if self.model_args.hybrid == 1:
#             logits = logits[0]
#             cls_logits = logits[1]

#         loss = None
#         if labels is not None:
#             if self.num_labels == 1:
#                 # Regression task
#                 loss_fct = nn.KLDivLoss(log_target=True)
#                 labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
#                                       (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
#                 loss = loss_fct(logits.view(-1, 2), labels)
#             else:

#                 if labels.shape == logits.shape:
#                     loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
#                                     labels, reduction="batchmean")
#                 else:
#                     loss_fct = nn.CrossEntropyLoss()

#                     loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

#         output = (logits,)
#         if self.num_labels == 1:
#             # Regression output
#             output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

#         return ((loss,) + output) if loss is not None else output



# # add by wjn
# # Prefix-tuning for Deberta
# class DebertaPrefixForPromptFinetuning(DebertaPreTrainedModel):

#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         #self.deberta = DebertaV2Model(config)

#         self.deberta = DebertaModel(config)
#         self.cls = DebertaOnlyMLMHead(config)

#         self.pooler = ContextPooler(config)
#         output_dim = self.pooler.output_dim

#         self.classifier = torch.nn.Linear(output_dim, self.num_labels)
#         drop_out = getattr(config, "cls_dropout", None)
#         drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

#         self.dropout = StableDropout(drop_out)

#         classification_list = [self.pooler, self.dropout,self.classifier]

#         self.classifier = nn.Sequential(*classification_list)
#         # self.cls = DebertaV2OnlyMLMHead(config)

#         self.map = nn.Linear(config.hidden_size, config.hidden_size)
#         self.init_weights()

#         if self.config.use_freezing:
#             self.deberta = freezer.freeze_lm(self.deberta)

#         self.pre_seq_len = config.pre_seq_len
#         self.n_layer = config.num_hidden_layers
#         self.n_head = config.num_attention_heads
#         self.n_embd = config.hidden_size // config.num_attention_heads

#         self.prefix_tokens = torch.arange(self.pre_seq_len).long()
#         self.prefix_encoder = PrefixEncoder(config)

#         # These attributes should be assigned once the model is initialized
#         self.model_args = None
#         self.data_args = None
#         self.label_word_list = torch.Tensor(self.config.label_word_list).long().to(self.bert.device)
#         self.K = 1
#         self.step_size=1e-5
#         # import pdb
#         # pdb.set_trace()
#         #self.step_size=config.step_size

#         # For regression
#         self.lb = None
#         self.ub = None


#         # For auto label search.
#         self.return_full_softmax = None

#     def freeze_backbone(self, use_freezing: bool=True):
#         if use_freezing:
#             self.deberta = freezer.freeze_lm(self.deberta)
#         else:
#             self.deberta = freezer.unfreeze_lm(self.deberta)

#     def get_prompt(self, batch_size):
#         prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.deberta.device)
#         past_key_values = self.prefix_encoder(prefix_tokens)
#         # bsz, seqlen, _ = past_key_values.shape
#         past_key_values = past_key_values.view(
#             batch_size,
#             self.pre_seq_len,
#             self.n_layer * 2,
#             self.n_head,
#             self.n_embd
#         )
#         past_key_values = self.dropout(past_key_values)
#         past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
#         return past_key_values


#     def get_constrast_loss(self,
#                     input_ids=None,
#                     attention_mask=None,
#                     mask_pos=None,
#                     labels=None,
#                     inputs_embeds=None):

#         self.cos = nn.CosineSimilarity(dim=-1)


#         _, sequence_mask_output_1 = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)
#         _, sequence_mask_output_2 = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

#         sequence_mask_output_1= self.lm_head.dense(sequence_mask_output_1)
#         sequence_mask_output_2 = self.lm_head.dense(sequence_mask_output_2)
#         # input_args = [input_ids, attention_mask, mask_pos, labels, None, 1]
#         # embed = self.forward(*input_args)
#         #
#         # vat_args = [input_ids, attention_mask, mask_pos, labels, embed, 2]
#         #
#         # adv_logits, outputs = self.forward(*vat_args)
#         #
#         # logit_mask = F.softmax(logits, dim=-1)[torch.arange(adv_logits.size(0)), labels] > 0.7
#         #
#         # outputs = outputs[logit_mask]
#         # seq_outputs = sequence_mask_output[logit_mask]
#         # new_label = labels[logit_mask]
#         # #
#         # #
#         # rand_perm = torch.randperm(outputs.size(0))
#         # rand_outputs = outputs[rand_perm, :]
#         # rand_label = new_label[rand_perm]
#         # pair_label = (new_label == rand_label).long()
#         #
#         # seq_outputs = self.map(seq_outputs)
#         # rand_outputs = self.map(rand_outputs)

#         pair_labels = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

#         # import  pdb
#         # pdb.set_trace()

#         contra_loss = self.contra_lc(sequence_mask_output_1.unsqueeze(1), sequence_mask_output_2.unsqueeze(0), pair_labels)

#         if torch.isnan(contra_loss):
#             return 0

#         return contra_loss

#     def embed_encode(self, input_ids):
#         embedding_output = self.deberta.embeddings.word_embeddings(input_ids)
#         return embedding_output

#     def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
#         batch_size = input_ids.size(0)

#         # add prefix for prompt-tuning
#         past_key_values = self.get_prompt(batch_size=batch_size)
#         prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.deberta.device)
#         attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

#         if mask_pos is not None:
#             mask_pos = mask_pos.squeeze()

#         # Encode everything

#         outputs = self.deberta(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             past_key_values=past_key_values,
#         )


#         # Get <mask> token representation
#         sequence_output, pooled_output = outputs[:2]
#         # sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
#         sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

#         # Logits over vocabulary tokens
#         prediction_mask_scores = self.cls(sequence_mask_output)

#         #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

#         # Exit early and only return mask logits.
#         if return_full_softmax:
#             return prediction_mask_scores

#         # Return logits for each label
#         logits = []
#         for label_id in range(len(self.label_word_list)):
#             logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
#         logits = torch.cat(logits, -1)

#         # Regression task
#         if self.config.num_labels == 1:
#             logsoftmax = nn.LogSoftmax(-1)
#             logits = logsoftmax(logits)  # Log prob of right polarity

#         if self.model_args.hybrid == 1:
#             cls_logits = self.classifier(sequence_output)
#             return (logits, cls_logits), sequence_mask_output

#         return logits, sequence_mask_output


#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             mask_pos=None,
#             labels=None,
#             inputs_embeds=None,
#             fwd_type=0,
#             block_flag=None,
#             return_dict=None,
#     ):

#         if fwd_type == 2:
#             assert inputs_embeds is not None
#             return self.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
#                                mask_pos=mask_pos, inputs_embeds=inputs_embeds)

#         elif fwd_type == 1:
#             return self.embed_encode(input_ids)



#         if (self.model_args.prompt_ptuning or self.model_args.prompt_prefix) and block_flag is not None:
#             inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)

#         logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)

#         if self.model_args.hybrid == 1:
#             logits = logits[0]
#             cls_logits = logits[1]

#         loss = None
#         if labels is not None:
#             if self.num_labels == 1:
#                 # Regression task
#                 loss_fct = nn.KLDivLoss(log_target=True)
#                 labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
#                                       (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
#                 loss = loss_fct(logits.view(-1, 2), labels)
#             else:

#                 if labels.shape == logits.shape:
#                     loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
#                                     labels, reduction="batchmean")
#                 else:
#                     loss_fct = nn.CrossEntropyLoss()

#                     loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

#         output = (logits,)
#         if self.num_labels == 1:
#             # Regression output
#             output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

#         if not return_dict:
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#         )




# class Debertav2ForPromptFinetuning(DebertaV2PreTrainedModel):
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#     _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.deberta = DebertaV2Model(config)

#         if self.config.use_freezing:
#             self.deberta = freezer.freeze_lm(self.deberta)
#         self.cls = DebertaV2OnlyMLMHead(config)

#         #self.deberta = DebertaModel(config)
#         #self.cls = DebertaOnlyMLMHead(config)

#         self.pooler = ContextPooler(config)
#         output_dim = self.pooler.output_dim

#         self.classifier = torch.nn.Linear(output_dim, self.num_labels)
#         drop_out = getattr(config, "cls_dropout", None)
#         drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

#         self.dropout = StableDropout(drop_out)

#         classification_list = [self.pooler, self.dropout,self.classifier]

#         self.classifier = nn.Sequential(*classification_list)
#         # self.cls = DebertaV2OnlyMLMHead(config)

#         self.map = nn.Linear(config.hidden_size, config.hidden_size)
#         self.init_weights()

#         # These attributes should be assigned once the model is initialized
#         self.model_args = None
#         self.data_args = None
#         self.label_word_list = torch.Tensor(self.config.label_word_list).long().to(self.bert.device)
#         self.K = 1
#         self.step_size=1e-5
#         # import pdb
#         # pdb.set_trace()
#         #self.step_size=config.step_size

#         # For regression
#         self.lb = None
#         self.ub = None

#         self.pre_seq_len = self.config.pre_seq_len
#         # For auto label search.
#         self.return_full_softmax = None

#     def freeze_backbone(self, use_freezing: bool=True):
#         if use_freezing:
#             self.deberta = freezer.freeze_lm(self.deberta)
#         else:
#             self.deberta = freezer.unfreeze_lm(self.deberta)

#     def embed_encode(self, input_ids):
#         embedding_output = self.deberta.embeddings.word_embeddings(input_ids)
#         return embedding_output

#     def encode(self, input_ids=None, attention_mask=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
#         batch_size = input_ids.size(0)

#         if mask_pos is not None:
#             mask_pos = mask_pos.squeeze()

#         # Encode everything
#         if inputs_embeds is None:
#             outputs =  self.deberta(
#                 input_ids,
#                 attention_mask=attention_mask
#             )
#         else:
#             outputs =  self.deberta(
#                 None,
#                 attention_mask=attention_mask,
#                 inputs_embeds=inputs_embeds
#             )


#         # Get <mask> token representation
#         sequence_output = outputs[0]
#         sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
#         sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]


#         # Logits over vocabulary tokens
#         prediction_mask_scores = self.cls(sequence_mask_output)

#         #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

#         # Exit early and only return mask logits.
#         if return_full_softmax:
#             return prediction_mask_scores

#         # Return logits for each label
#         logits = []
#         for label_id in range(len(self.label_word_list)):
#             logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
#         logits = torch.cat(logits, -1)

#         # Regression task
#         if self.config.num_labels == 1:
#             logsoftmax = nn.LogSoftmax(-1)
#             logits = logsoftmax(logits)  # Log prob of right polarity

#         return logits, sequence_mask_output


#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         mask_pos=None,
#         labels=None,
#         inputs_embeds=None,
#         fwd_type=0,
#         block_flag=None,
#         return_dict=None
#     ):
#         if fwd_type == 2:
#             assert inputs_embeds is not None
#             return self.encode(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos, inputs_embeds=inputs_embeds)

#         elif fwd_type == 1:
#             return self.embed_encode(input_ids)

#         logits, sequence_mask_output = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

#         loss = None


#         if labels is not None:
#             if self.num_labels == 1:
#                 # Regression task
#                 loss_fct = nn.KLDivLoss(log_target=True)
#                 labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
#                 loss = loss_fct(logits.view(-1, 2), labels)
#             else:

#                 if labels.shape == logits.shape:
#                     loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
#                                     labels, reduction="batchmean")
#                 else:
#                     loss_fct = nn.CrossEntropyLoss()

#                     loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#                     if self.model_args.hybrid == 1:
#                         cls_loss = loss_fct(cls_logits.view(-1, cls_logits.size(-1)), labels.view(-1))
#                         loss = loss + cls_loss

#         output = (logits,)
#         if self.num_labels == 1:
#             # Regression output
#             output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

#         if not return_dict:
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#         )


# class Debertav2PrefixForPromptFinetuning(DebertaV2PreTrainedModel):
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#     _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels
#         self.deberta = DebertaV2Model(config)
#         self.cls = DebertaV2OnlyMLMHead(config)

#         #self.deberta = DebertaModel(config)
#         #self.cls = DebertaOnlyMLMHead(config)

#         self.pooler = ContextPooler(config)
#         output_dim = self.pooler.output_dim

#         self.classifier = torch.nn.Linear(output_dim, self.num_labels)
#         drop_out = getattr(config, "cls_dropout", None)
#         drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

#         self.dropout = StableDropout(drop_out)

#         classification_list = [self.pooler, self.dropout,self.classifier]

#         self.classifier = nn.Sequential(*classification_list)
#         # self.cls = DebertaV2OnlyMLMHead(config)

#         self.map = nn.Linear(config.hidden_size, config.hidden_size)
#         self.init_weights()

#         if self.config.use_freezing:
#             self.deberta = freezer.freeze_lm(self.deberta)

#         self.pre_seq_len = config.pre_seq_len
#         self.n_layer = config.num_hidden_layers
#         self.n_head = config.num_attention_heads
#         self.n_embd = config.hidden_size // config.num_attention_heads

#         self.prefix_tokens = torch.arange(self.pre_seq_len).long()
#         self.prefix_encoder = PrefixEncoder(config)

#         # These attributes should be assigned once the model is initialized
#         self.model_args = None
#         self.data_args = None
#         self.label_word_list = torch.Tensor(self.config.label_word_list).long().to(self.bert.device)
#         self.K = 1
#         self.step_size=1e-5
#         # import pdb
#         # pdb.set_trace()
#         #self.step_size=config.step_size

#         # For regression
#         self.lb = None
#         self.ub = None


#         # For auto label search.
#         self.return_full_softmax = None

#     def freeze_backbone(self, use_freezing: bool=True):
#         if use_freezing:
#             self.deberta = freezer.freeze_lm(self.deberta)
#         else:
#             self.deberta = freezer.unfreeze_lm(self.deberta)

#     def get_prompt(self, batch_size):
#         prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.deberta.device)
#         past_key_values = self.prefix_encoder(prefix_tokens)
#         # bsz, seqlen, _ = past_key_values.shape
#         past_key_values = past_key_values.view(
#             batch_size,
#             self.pre_seq_len,
#             self.n_layer * 2,
#             self.n_head,
#             self.n_embd
#         )
#         past_key_values = self.dropout(past_key_values)
#         past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
#         return past_key_values


#     def embed_encode(self, input_ids):
#         embedding_output = self.deberta.embeddings.word_embeddings(input_ids)
#         return embedding_output

#     def encode(self, input_ids=None, attention_mask=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
#         batch_size = input_ids.size(0)

#         # add prefix for prompt-tuning
#         past_key_values = self.get_prompt(batch_size=batch_size)
#         prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.deberta.device)
#         attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)


#         if mask_pos is not None:
#             mask_pos = mask_pos.squeeze()

#         # Encode everything
#         outputs = self.deberta(
#             input_ids,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#         )


#         # Get <mask> token representation
#         sequence_output = outputs[0]
#         # sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
#         sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]


#         # Logits over vocabulary tokens
#         prediction_mask_scores = self.cls(sequence_mask_output)

#         #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

#         # Exit early and only return mask logits.
#         if return_full_softmax:
#             return prediction_mask_scores

#         # Return logits for each label
#         logits = []
#         for label_id in range(len(self.label_word_list)):
#             logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
#         logits = torch.cat(logits, -1)

#         # Regression task
#         if self.config.num_labels == 1:
#             logsoftmax = nn.LogSoftmax(-1)
#             logits = logsoftmax(logits)  # Log prob of right polarity

#         return logits, sequence_mask_output


#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         mask_pos=None,
#         labels=None,
#         inputs_embeds=None,
#         fwd_type=0,
#         block_flag=None,
#         return_dict=None,
#     ):
#         if fwd_type == 2:
#             assert inputs_embeds is not None
#             return self.encode(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos, inputs_embeds=inputs_embeds)

#         elif fwd_type == 1:
#             return self.embed_encode(input_ids)

#         logits, sequence_mask_output = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

#         loss = None


#         if labels is not None:
#             if self.num_labels == 1:
#                 # Regression task
#                 loss_fct = nn.KLDivLoss(log_target=True)
#                 labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
#                 loss = loss_fct(logits.view(-1, 2), labels)
#             else:

#                 if labels.shape == logits.shape:
#                     loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
#                                     labels, reduction="batchmean")
#                 else:
#                     loss_fct = nn.CrossEntropyLoss()

#                     loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#                     if self.model_args.hybrid == 1:
#                         cls_loss = loss_fct(cls_logits.view(-1, cls_logits.size(-1)), labels.view(-1))
#                         loss = loss + cls_loss

#         output = (logits,)
#         if self.num_labels == 1:
#             # Regression output
#             output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

#         if not return_dict:
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss,
#             logits=logits,
#         )
