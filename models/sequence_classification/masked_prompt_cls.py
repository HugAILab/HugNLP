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
# from processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping
import logging
from models.basic_modules.adapter import RobertaAdaModel, BertAdaModel
import os
from models.basic_modules.prefix_encoder import PrefixEncoder
from tools.model_utils.parameter_freeze import ParameterFreeze

freezer = ParameterFreeze()


logger = logging.getLogger(__name__)




### Notes! If you set user-defined datasets, you must add some settings first.


num_labels_mapping = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "snli": 3,
    "movie_rationales": 2,
    "sst-5": 5,
    "subj": 2,
    "trec": 6,
    "cr": 2,
    "mpqa": 2,
    "boolq": 2,
    "cb": 3, 
    "ag_news": 4,
    "yelp_polarity": 2,
}

output_modes_mapping = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mnli-mm-clue": "classification",
    "mrpc": "classification",
    "sst2": "classification",
    "sst-2-clue": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qqp-clue": "classification",
    "qnli": "classification",
    "rte": "classification",
    "rte-clue": "classification",
    "wnli": "classification",
    "snli": "classification",
    "movie_rationales": "classification",
    "mr-clue": "classification",
    "sst-5": "classification",
    "subj": "classification",
    "subj-clue": "classification",
    "trec": "classification",
    "cr": "classification",
    "mpqa": "classification",
    "mpqa-clue": "classification",
    "boolq": "classification",
    "cb": "classification",
    "ag_news": "classification",
    "yelp_polarity": "classification"
}

# For regression task only: median
median_mapping = {
    "sts-b": 2.5
}

bound_mapping = {
    "sts-b": (0, 5)
}




# Training the model with prompt and verbalizer
class LMForPromptFinetuning(BertPreTrainedModel):
    def __init__(self, config, model_args, data_args):
        super().__init__(config)
        self.model_args = model_args
        self.data_args = data_args
        self.config = config
        # Create config
        num_labels = num_labels_mapping[data_args.dataset_name]
        self.num_labels = num_labels
        config.adapter_dim = model_args.adapter_dim
        try:
            config.adapter_alpha = model_args.adapter_alpha
        except:
            config.adapter_alpha = 32
        config.adapter_choice = model_args.adapter_choice
        self.pre_seq_len = self.model_args.pre_seq_len
        config.pre_seq_len = self.pre_seq_len
        self.config = config

        if config.model_type == 'roberta':
            if model_args.prompt_prefix:
                model_fn = RobertPrefixForPromptFinetuning
            elif model_args.prompt_ptuning:
                model_fn = RobertaForPromptFinetuning
            elif model_args.prompt_adapter:
                model_fn = RobertaAdapterForPromptFinetuning
            else:
                model_fn = RobertaForPromptFinetuning

        elif config.model_type == 'bert':
            if model_args.prompt_prefix:
                model_fn = BertPrefixForPromptFinetuning
            elif model_args.prompt_ptuning:
                model_fn = BertForPromptFinetuning
            elif model_args.prompt_adapter:
                model_fn = BertAdapterForPromptFinetuning
            else:
                model_fn = BertForPromptFinetuning

        elif config.model_type == 'deberta':
            if model_args.prompt_prefix:
                model_fn = DebertPrefixForPromptFinetuning
            elif model_args.prompt_ptuning:
                model_fn = DebertaForPromptFinetuning
            elif model_args.prompt_adapter:
                pass
            else:
                model_fn = DebertaForPromptFinetuning

        elif config.model_type == 'deberta-v2':
            if model_args.prompt_prefix:
                pass
            elif model_args.prompt_ptuning:
                model_fn = Debertav2ForPromptFinetuning
            elif model_args.prompt_adapter:
                pass
            else:
                model_fn = Debertav2ForPromptFinetuning

        # elif config.model_type == 't5':
        #     if model_args.prompt_prefix:
        #         pass
        #     elif model_args.prompt_ptuning:
        #         self.lm_model = T5ForPromptFinetuning(config)
        #     elif model_args.prompt_adapter:
        #         pass
        #     else:
        #         self.lm_model = T5ForPromptFinetuning(config)

        else:
            raise NotImplementedError


        if config.model_type == 't5':
            self.lm_model.T5 =  self.lm_model.T5.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

        else:
            self.lm_model = model_fn.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
        
        if config.model_type == "roberta":
            self.embeddings = self.lm_model.roberta.embeddings
        elif config.model_type == "bert":
            self.embeddings = self.lm_model.bert.embeddings
        elif config.model_type in ["deberta", "deberta-v2"]:
            self.embeddings = self.lm_model.deberta.embeddings
        elif config.model_type == "t5":
            self.embeddings = self.lm_model.T5.embeddings


        # Pass dataset and argument information to the model
        if model_args.prompt_prefix or model_args.prompt_ptuning or model_args.prompt_adapter or model_args.prompt_only:
            self.lm_model.label_word_list = torch.tensor(data_args.label_word_list).long().cuda()
        else:
            raise RuntimeError("You must choose prompt_prefix or prompt_ptuning or prompt_adapter or prompt_only.")
        
        if output_modes_mapping[data_args.dataset_name] == 'regression':
            # lower / upper bounds
            self.lm_model.lb, self.lm_model.ub = bound_mapping[data_args.dataset_name]
        
        self.lm_model.model_args = model_args
        self.lm_model.data_args = data_args
        self.hidden_size = config.hidden_size
        
        # edit by wjn
        if self.model_args.prompt_ptuning:
            self.prompt_embeddings = torch.nn.Embedding(self.pre_seq_len, self.hidden_size)
        else:
            self.prompt_embeddings = None
        # self.data_args.continuous_prompt = self.model_args.pre_seq_len

        if self.model_args.prompt_adapter and self.model_args.adapter_choice != 'none':
            try:
                std = self.model_args.adapter_init_std
            except:
                std = 0.0002
            self.init_adapter(std=std)

        self.prompt_encoder = None
        # add by wjn
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)
        # if self.model_args.prompt_ptuning or self.model_args.prompt_prefix:
        #     self.init_embedding()

    def init_adapter(self, std):
        with torch.no_grad():
            for name, param in self.lm_model.named_parameters():
                init_value = 0
                if 'adapter_proj' in name:
                    if self.model_args.adapter_choice == 'simple':
                        init_value = torch.eye(param.size(0))
                    if std > 0:
                        init_value += torch.normal(0, std, size=param.size())
                    param.copy_(init_value)

    def init_embedding(self):

        rand_id = torch.randint(100, self.config.vocab_size, (self.model_args.pre_seq_len,)).long()
        rand_emb = self.lm_model.embed_encode(rand_id)
        self.prompt_embeddings = self.prompt_embeddings.from_pretrained(rand_emb, freeze=False)


    def embed_encode(self, input_ids):
        embedding_output = self.lm_model.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        # import pdb
        # pdb.set_trace()
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        if inputs_embeds is None:
            outputs = self.lm_model(
                input_ids,
                attention_mask=attention_mask
            )
        else:
            outputs = self.lm_model(
                None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds
            )

        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # sequence_mask_output = self.lm_head.dense(sequence_mask_output)

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

        # if self.model_args.hybrid == 1:
        #     cls_logits = self.classifier(sequence_output)
        #     return (logits, cls_logits), sequence_mask_output

        return logits, sequence_mask_output
    
    # add by wjn
    # insert soft prompt in input
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.lm_model.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    # add soft prompt via block_flag
    def generate_continuous_prompt_inputs(self, input_ids, block_flag):

        inputs_embeds = self.lm_model.embed_encode(input_ids)
        bz = inputs_embeds.shape[0]

        try:
            replace_embeds = self.prompt_embeddings(
                torch.LongTensor(list(range(self.model_args.pre_seq_len))).to(inputs_embeds.device))
        except:
            import pdb
            pdb.set_trace()
            replace_embeds = self.prompt_embeddings(
                torch.LongTensor(list(range(self.model_args.pre_seq_len))))

        replace_embeds = replace_embeds.unsqueeze(0)  # [batch_size, prompt_length, embed_size]


        if self.prompt_encoder is not None:
            replace_embeds = self.prompt_encoder(replace_embeds)

        blocked_indices = (block_flag == 1).nonzero(as_tuple=False).reshape((bz, self.model_args.pre_seq_len, 2))[:, :, 1]

        for bidx in range(bz):
            for i in range(blocked_indices.shape[1]):
                inputs_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[:, i, :].squeeze()

        return inputs_embeds

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            mask_pos=None,
            labels=None,
            inputs_embeds=None,
            fwd_type=0,
            block_flag=None,
            *args,
            **kwargs
    ):
        # print("mask_pos=", mask_pos)
        # print("fwd_type=", fwd_type) # 0
        batch_size = input_ids.shape[0]

        if 't5' in self.config.model_type:
            logits, sequence_mask_output = self.lm_model.encode(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        else:

            if fwd_type == 2:
                assert inputs_embeds is not None
                if token_type_ids is not None:
                    return self.lm_model.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mask_pos=mask_pos,
                                                inputs_embeds=inputs_embeds)
                else:
                    return self.lm_model.encode(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos,
                                                inputs_embeds=inputs_embeds)

            elif fwd_type == 1:
                return self.lm_model.embed_encode(input_ids)


            # print("block_flag=", block_flag) # None

            # if (self.model_args.prompt_ptuning or self.model_args.prompt_prefix) and block_flag is not None and block_flag[0] is not None:
            #     inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)
            # print("inputs_embeds=", inputs_embeds)
            if self.model_args.prompt_ptuning:

                if inputs_embeds is None:
                    raw_embedding = self.embeddings(
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                    )

                    prompts = self.get_prompt(batch_size=batch_size)
                    inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)

                    prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.lm_model.device)
                    attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)


            if fwd_type == 3:
                if token_type_ids is not None:
                    prediction_mask_scores = self.lm_model.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mask_pos=mask_pos,
                                                inputs_embeds=inputs_embeds, return_full_softmax=True)
                else:
                    prediction_mask_scores = self.lm_model.encode(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos, inputs_embeds=inputs_embeds, return_full_softmax=True)
                if labels is not None:
                    return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
                return prediction_mask_scores

            if token_type_ids is not None:
                logits, sequence_mask_output = self.lm_model.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mask_pos=mask_pos,
                                                inputs_embeds=inputs_embeds)
            else:
                logits, sequence_mask_output = self.lm_model.encode(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos, inputs_embeds=inputs_embeds)

            if fwd_type == 4:
                return logits, sequence_mask_output

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                                      (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:


                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction='batchmean')
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        return ((loss,) + output) if loss is not None else output


    def from_pretrained(self, pretrained_model_name_or_path, *model_args, **kwargs):

        self.lm_model = self.lm_model.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        if self.data_args.prompt:
            self.lm_model.label_word_list = torch.tensor(self.data_args.label_word_list).long().cuda()
        if output_modes_mapping[self.data_args.dataset_name] == 'regression':
            # lower / upper bounds
            self.lm_model.lb, self.lm_model.ub = bound_mapping[self.data_args.dataset_name]
        self.lm_model.model_args = self.model_args
        self.lm_model.data_args = self.data_args

        return self

    def load_model(self, checkpoint):

        if os.path.isfile(checkpoint):
            model_state_dict = torch.load(checkpoint)
            self.load_state_dict(model_state_dict, strict=False)

# Only prompt for BERT
class PromptBertForSequenceClassification(LMForPromptFinetuning):
    def __init__(self, config, model_args, data_args):
        config.model_type = "bert"
        model_args.prompt_only = True
        super().__init__(config, model_args, data_args)

# Prefix-tuning for BERT
class PromptBertPrefixForSequenceClassification(LMForPromptFinetuning):
    def __init__(self, config, model_args, data_args):
        config.model_type = "bert"
        model_args.prompt_prefix = True
        super().__init__(config, model_args, data_args)

# P-tuning for BERT
class PromptBertPtuningForSequenceClassification(LMForPromptFinetuning):
    def __init__(self, config, model_args, data_args):
        config.model_type = "bert"
        model_args.prompt_ptuning = True
        super().__init__(config, model_args, data_args)

# Adapter for BERT
class PromptBertAdapterForSequenceClassification(LMForPromptFinetuning):
    def __init__(self, config, model_args, data_args):
        config.model_type = "bert"
        model_args.prompt_adapter = True
        super().__init__(config, model_args, data_args)

# Prefix-tuning for RoBERTa
class PromptRobertaPrefixForSequenceClassification(LMForPromptFinetuning):
    def __init__(self, config, model_args, data_args):
        config.model_type = "roberta"
        model_args.prompt_prefix = True
        super().__init__(config, model_args, data_args)

# Only prompt for Roberta
class PromptRobertaForSequenceClassification(LMForPromptFinetuning):
    def __init__(self, config, model_args, data_args):
        config.model_type = "roberta"
        model_args.prompt_only = True
        super().__init__(config, model_args, data_args)

# P-tuning for Roberta
class PromptRobertaPtuningForSequenceClassification(LMForPromptFinetuning):
    def __init__(self, config, model_args, data_args):
        config.model_type = "roberta"
        model_args.prompt_ptuning = True
        super().__init__(config, model_args, data_args)

# Adapter for RoBERTa
class PromptRobertaAdapterForSequenceClassification(LMForPromptFinetuning):
    def __init__(self, config, model_args, data_args):
        config.model_type = "roberta"
        model_args.prompt_adapter = True
        super().__init__(config, model_args, data_args)

# Only prompt for DeBERTa
class PromptDebertaForSequenceClassification(LMForPromptFinetuning):
    def __init__(self, config, model_args, data_args):
        config.model_type = "deberta"
        model_args.prompt_only = True
        super().__init__(config, model_args, data_args)

# Prefix-tuning for DeBERTa
class PromptDebertaPrefixForSequenceClassification(LMForPromptFinetuning):
    def __init__(self, config, model_args, data_args):
        config.model_type = "deberta"
        model_args.prompt_prefix = True
        super().__init__(config, model_args, data_args)

# P-tuning for DeBERTa
class PromptDebertaPtuningForSequenceClassification(LMForPromptFinetuning):
    def __init__(self, config, model_args, data_args):
        config.model_type = "deberta"
        model_args.prompt_ptuning = True
        super().__init__(config, model_args, data_args)

# # Adapter for Deberta
# class PromptDebertaAdapterForSequenceClassification(LMForPromptFinetuning):
#     def __init__(self, config, model_args, data_args):
#         config.model_type = "deberta"
#         model_args.prompt_adapter = True
#         super().__init__(config, model_args, data_args)

# Only prompt for BERT
class PromptDebertav2ForSequenceClassification(LMForPromptFinetuning):
    def __init__(self, config, model_args, data_args):
        config.model_type = "deberta-v2"
        model_args.prompt_only = True
        super().__init__(config, model_args, data_args)

# Prefix-tuning for DeBERTa-v2
class PromptDebertav2PrefixForSequenceClassification(LMForPromptFinetuning):
    def __init__(self, config, model_args, data_args):
        config.model_type = "deberta-v2"
        model_args.prompt_prefix = True
        super().__init__(config, model_args, data_args)

# P-tuning for DeBERTa-v2
class PromptDebertav2PtuningForSequenceClassification(LMForPromptFinetuning):
    def __init__(self, config, model_args, data_args):
        config.model_type = "deberta-v2"
        model_args.prompt_ptuning = True
        super().__init__(config, model_args, data_args)

# # Adapter for Deberta-V2
# class PromptDebertav2AdapterForSequenceClassification(LMForPromptFinetuning):
#     def __init__(self, config, model_args, data_args):
#         config.model_type = "deberta-v2"
#         model_args.prompt_adapter = True
#         super().__init__(config, model_args, data_args)


# p-tuning for bert
class BertForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        if self.config.use_pe:
            self.bert = freezer.freeze_lm(self.bert)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None
        self.pre_seq_len = self.config.pre_seq_len
        # For label search.
        self.return_full_softmax = None
    
    def freeze_backbone(self, use_pe: bool=True):
        if use_pe:
            self.bert = freezer.freeze_lm(self.bert)
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
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

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

        # if self.model_args.hybrid == 1:
        #     cls_logits = self.classifier(sequence_output)
        #     return (logits, cls_logits), sequence_mask_output

        return logits, sequence_mask_output


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        fwd_type=0,
        block_flag=None,
        return_dict=None,
    ):
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mask_pos=mask_pos, inputs_embeds=inputs_embeds)

        elif fwd_type == 1:
            return self.embed_encode(input_ids)

        if (self.model_args.prompt_ptuning or self.model_args.prompt_prefix) and block_flag is not None:
            inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)

        if self.model_args.hybrid == 1:

            logits = logits[0]
            cls_logits = logits[1]

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
                                    labels, reduction='batchmean')
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


# add by wjn
# Prefix-tuning for BERT
class BertPrefixForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

        if self.config.use_pe:
            self.bert = freezer.freeze_lm(self.bert)
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None
    
    def freeze_backbone(self, use_pe: bool=True):
        if use_pe:
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

        #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

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

        # if self.model_args.hybrid == 1:
        #     cls_logits = self.classifier(sequence_output)
        #     return (logits, cls_logits), sequence_mask_output

        return logits, sequence_mask_output


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        fwd_type=0,
        block_flag=None,
        return_dict=None,
    ):
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mask_pos=mask_pos, inputs_embeds=inputs_embeds)

        elif fwd_type == 1:
            return self.embed_encode(input_ids)

        if (self.model_args.prompt_ptuning or self.model_args.prompt_prefix) and block_flag is not None:
            inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)

        if self.model_args.hybrid == 1:

            logits = logits[0]
            cls_logits = logits[1]

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
                                    labels, reduction='batchmean')
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

# Adapter for Bert
class BertAdapterForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertAdaModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        if self.config.use_pe:
            self.bert = freezer.freeze_lm_component(self.bert, "adapter")

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None
    
    def freeze_backbone(self, use_pe: bool=True):
        if use_pe:
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

        #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

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

        # if self.model_args.hybrid == 1:
        #     cls_logits = self.classifier(sequence_output)
        #     return (logits, cls_logits), sequence_mask_output

        return logits, sequence_mask_output


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        fwd_type=0,
        block_flag=None,
        return_dict=None,
    ):
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mask_pos=mask_pos, inputs_embeds=inputs_embeds)

        elif fwd_type == 1:
            return self.embed_encode(input_ids)

        if (self.model_args.prompt_ptuning or self.model_args.prompt_prefix) and block_flag is not None:
            inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)

        if self.model_args.hybrid == 1:

            logits = logits[0]
            cls_logits = logits[1]

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
                                    labels, reduction='batchmean')
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



class RobertaForPromptFinetuning(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        # self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.hidden_size = config.hidden_size

        # self.map = nn.Linear(config.hidden_size, config.hidden_size)

        if self.config.use_pe:
            self.roberta = freezer.freeze_lm(self.roberta)

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.K = 1
        self.step_size=1e-5

        # For regression
        self.lb = None
        self.ub = None

        self.tokenizer = None

        self.prompt_embeddings = None
        self.lstm_head = None
        self.mlp_head = None
        self.mlp = None

        # For auto label search.
        self.return_full_softmax = None
        self.pre_seq_len = self.config.pre_seq_len
        #self.init_weights()
        # else:
        #     raise ValueError('unknown prompt_encoder_type.')

    def freeze_backbone(self, use_pe: bool=True):
        if use_pe:
            self.roberta = freezer.freeze_lm(self.roberta)
        else:
            self.roberta = freezer.unfreeze_lm(self.roberta)

    def embed_encode(self, input_ids):
        embedding_output = self.roberta.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        if inputs_embeds is None:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask
            )
        else:
            outputs = self.roberta(
                None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds
            )


        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []

        for label_id in range(len(self.label_word_list)):
            # print("label_id=", label_id)
            # print("self.label_word_list=", self.label_word_list)
            # print("prediction_mask_scores.shape=", prediction_mask_scores.shape)
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        # if self.model_args.hybrid == 1:
        #     cls_logits = self.classifier(sequence_output)
        #     return (logits, cls_logits), sequence_mask_output


        return logits, sequence_mask_output

    def generate_continuous_prompt_inputs(self, input_ids, block_flag):

        inputs_embeds = self.embed_encode(input_ids)
        bz = inputs_embeds.shape[0]

        try:
            replace_embeds = self.prompt_embeddings(
                torch.LongTensor(list(range(1))).to(inputs_embeds.device))
        except:
            import pdb
            pdb.set_trace()
            replace_embeds = self.prompt_embeddings(
                torch.LongTensor(list(range(1))))

        replace_embeds = replace_embeds.unsqueeze(0)  # [batch_size, prompt_length, embed_size]

        # if self.model_args.prompt_encoder_type == "lstm":
        #     replace_embeds = self.lstm_head(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
        #     if self.prompt_length == 1:
        #         replace_embeds = self.mlp_head(replace_embeds)
        #     else:
        #         replace_embeds = self.mlp_head(replace_embeds).squeeze()

        # elif self.model_args.prompt_encoder_type == "mlp":
        replace_embeds = self.mlp(replace_embeds)
        # else:
        #     raise ValueError("unknown prompt_encoder_type.")

        blocked_indices = (block_flag == 1).nonzero(as_tuple=False).reshape((bz, self.model_args.prompt_length, 2))[:, :, 1]

        for bidx in range(bz):
            for i in range(blocked_indices.shape[1]):
                inputs_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

        return inputs_embeds



    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        fwd_type=0,
        block_flag=None,
        return_dict=None
    ):
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos, inputs_embeds=inputs_embeds)

        elif fwd_type == 1:
            return self.embed_encode(input_ids)

        if (self.model_args.prompt_ptuning or self.model_args.prompt_prefix) and block_flag is not None:
            inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

        if self.model_args.hybrid == 1:

            logits = logits[0]
            cls_logits = logits[1]

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
                                    labels, reduction='batchmean')
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



# add by wjn
# Prefix-tuning for RoBERTa
class RobertPrefixForPromptFinetuning(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

        if self.config.use_pe:
            self.roberta = freezer.freeze_lm(self.roberta)
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For label search.
        self.return_full_softmax = None
    
    def freeze_backbone(self, use_pe: bool=True):
        if use_pe:
            self.roberta = freezer.freeze_lm(self.roberta)
        else:
            self.roberta = freezer.unfreeze_lm(self.roberta)
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
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
        embedding_output = self.roberta.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        batch_size = input_ids.size(0)

        # add prefix for prompt-tuning
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        # print("input_ids.shape=", input_ids.shape)
        # print("token_type_ids.shape=", token_type_ids.shape)
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            past_key_values=past_key_values,
        )


        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        # print("sequence_output.shape=", sequence_output.shape)
        # sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        # print("sequence_output.size(0))=", sequence_output.size(0))
        # print("mask_pos.shape=", mask_pos.shape)
        # print("input_ids[mask_pos]=", input_ids[torch.arange(sequence_output.size(0)), mask_pos])
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        # print("prediction_mask_scores.shape=", prediction_mask_scores.shape)
        # print("self.label_word_list=", self.label_word_list)
        for label_id in range(len(self.label_word_list)):
            # print("label_id=", label_id)
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)
        # print("logits=", logits)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        # if self.model_args.hybrid == 1:
        #     cls_logits = self.classifier(sequence_output)
        #     return (logits, cls_logits), sequence_mask_output

        return logits, sequence_mask_output


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        fwd_type=0,
        block_flag=None,
        return_dict=None,
    ):
        # print("token_type_ids.shape=", token_type_ids.shape)
        # print("token_type_ids=", token_type_ids)
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, mask_pos=mask_pos, inputs_embeds=inputs_embeds)

        elif fwd_type == 1:
            return self.embed_encode(input_ids)

        if (self.model_args.prompt_ptuning or self.model_args.prompt_prefix) and block_flag is not None:
            inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)

        if self.model_args.hybrid == 1:

            logits = logits[0]
            cls_logits = logits[1]

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
                                    labels, reduction='batchmean')
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

# Adapter for RoBERTa
class RobertaAdapterForPromptFinetuning(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaAdaModel(config)
        # self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.hidden_size = config.hidden_size

        # self.map = nn.Linear(config.hidden_size, config.hidden_size)


        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.K = 1
        self.step_size=1e-5

        # For regression
        self.lb = None
        self.ub = None

        self.tokenizer = None

        self.prompt_embeddings = None
        self.lstm_head = None
        self.mlp_head = None
        self.mlp = None

        # For auto label search.
        self.return_full_softmax = None

        self.init_weights()
        if self.config.use_pe:
            self.roberta = freezer.freeze_lm_component(self.roberta, "adapter")
        # else:
        #     raise ValueError('unknown prompt_encoder_type.')
    
    def freeze_backbone(self, use_pe: bool=True):
        if use_pe:
            self.roberta = freezer.freeze_lm_component(self.roberta, "adapter")
        else:
            self.roberta = freezer.unfreeze_lm(self.roberta)


    def embed_encode(self, input_ids):
        embedding_output = self.roberta.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        if inputs_embeds is None:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask
            )
        else:
            outputs = self.roberta(
                None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds
            )

        # print("mask_pos.shape=", mask_pos.shape)
        # Get <mask> token representation
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]


        # print("sequence_mask_output.shape=", sequence_mask_output.shape) # torch.Size([4, 1, 128, 1024])
        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

        # Exit early and only return mask logits.
        if return_full_softmax:
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        # print("self.label_word_list=", self.label_word_list) # e.g., tensor([ 7447, 35297], device='cuda:0')
        for label_id in range(len(self.label_word_list)):
            # print("label_id=", label_id) # e.g., 0
            # print("self.label_word_list[label_id]=", self.label_word_list[label_id]) # e.g., tensor(7447, device='cuda:0')
            # print("prediction_mask_scores.shape=", prediction_mask_scores.shape) # [4, 1, 128, 50265] [bz, 1, seq_len, vocab_size]
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)
        # print("logits.shape=", logits.shape) # torch.Size([4, 1, 128, 2])
        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits)  # Log prob of right polarity

        # if self.model_args.hybrid == 1:
        #     cls_logits = self.classifier(sequence_output)
        #     return (logits, cls_logits), sequence_mask_output


        return logits, sequence_mask_output

    def generate_continuous_prompt_inputs(self, input_ids, block_flag):

        inputs_embeds = self.embed_encode(input_ids)
        bz = inputs_embeds.shape[0]

        try:
            replace_embeds = self.prompt_embeddings(
                torch.LongTensor(list(range(1))).to(inputs_embeds.device))
        except:
            import pdb
            pdb.set_trace()
            replace_embeds = self.prompt_embeddings(
                torch.LongTensor(list(range(1))))

        replace_embeds = replace_embeds.unsqueeze(0)  # [batch_size, prompt_length, embed_size]

        # if self.model_args.prompt_encoder_type == "lstm":
        #     replace_embeds = self.lstm_head(replace_embeds)[0]  # [batch_size, seq_len, 2 * hidden_dim]
        #     if self.prompt_length == 1:
        #         replace_embeds = self.mlp_head(replace_embeds)
        #     else:
        #         replace_embeds = self.mlp_head(replace_embeds).squeeze()

        # elif self.model_args.prompt_encoder_type == "mlp":
        replace_embeds = self.mlp(replace_embeds)
        # else:
        #     raise ValueError("unknown prompt_encoder_type.")

        blocked_indices = (block_flag == 1).nonzero(as_tuple=False).reshape((bz, self.model_args.prompt_length, 2))[:, :, 1]

        for bidx in range(bz):
            for i in range(blocked_indices.shape[1]):
                inputs_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

        return inputs_embeds



    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        fwd_type=0,
        block_flag=None,
        return_dict=None
    ):
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos, inputs_embeds=inputs_embeds)

        elif fwd_type == 1:
            return self.embed_encode(input_ids)

        if (self.model_args.prompt_ptuning or self.model_args.prompt_prefix) and block_flag is not None:
            inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

        if self.model_args.hybrid == 1:

            logits = logits[0]
            cls_logits = logits[1]

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
                                    labels, reduction='batchmean')
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


class DebertaForPromptFinetuning(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        #self.deberta = DebertaV2Model(config)

        self.deberta = DebertaModel(config)
        self.cls = DebertaOnlyMLMHead(config)

        if self.config.use_pe:
            self.deberta = freezer.freeze_lm(self.deberta)

        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = torch.nn.Linear(output_dim, self.num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.dropout = StableDropout(drop_out)

        classification_list = [self.pooler, self.dropout,self.classifier]

        self.classifier = nn.Sequential(*classification_list)
        # self.cls = DebertaV2OnlyMLMHead(config)

        self.map = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.K = 1
        self.step_size=1e-5
        # import pdb
        # pdb.set_trace()
        #self.step_size=config.step_size

        # For regression
        self.lb = None
        self.ub = None

        self.pre_seq_len = self.config.pre_seq_len
        # For auto label search.
        self.return_full_softmax = None
    
    def freeze_backbone(self, use_pe: bool=True):
        if use_pe:
            self.deberta = freezer.freeze_lm(self.deberta)
        else:
            self.deberta = freezer.unfreeze_lm(self.deberta)



    def embed_encode(self, input_ids):
        embedding_output = self.deberta.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None,
               return_full_softmax=False):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()


        # Encode everything
        if inputs_embeds is None:
            outputs = self.deberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.deberta(
                None,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds
            )

        # Get <mask> token representation
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]

        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        # sequence_mask_output = self.lm_head.dense(sequence_mask_output)

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

        if self.model_args.hybrid == 1:
            cls_logits = self.classifier(sequence_output)
            return (logits, cls_logits), sequence_mask_output

        return logits, sequence_mask_output

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            mask_pos=None,
            labels=None,
            inputs_embeds=None,
            fwd_type=0,
            block_flag=None
    ):
        
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               mask_pos=mask_pos, inputs_embeds=inputs_embeds)

        elif fwd_type == 1:
            return self.embed_encode(input_ids)



        if (self.model_args.prompt_ptuning or self.model_args.prompt_prefix) and block_flag is not None:
            inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)

        if self.model_args.hybrid == 1:
            logits = logits[0]
            cls_logits = logits[1]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                                      (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction='batchmean')
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)

        return ((loss,) + output) if loss is not None else output



# add by wjn
# Prefix-tuning for Deberta
class DebertaPrefixForPromptFinetuning(DebertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        #self.deberta = DebertaV2Model(config)

        self.deberta = DebertaModel(config)
        self.cls = DebertaOnlyMLMHead(config)

        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = torch.nn.Linear(output_dim, self.num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.dropout = StableDropout(drop_out)

        classification_list = [self.pooler, self.dropout,self.classifier]

        self.classifier = nn.Sequential(*classification_list)
        # self.cls = DebertaV2OnlyMLMHead(config)

        self.map = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

        if self.config.use_pe:
            self.deberta = freezer.freeze_lm(self.deberta)

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.K = 1
        self.step_size=1e-5
        self.adv_lc = SymKlCriterion()
        self.contra_lc = ContrastiveLoss()
        # import pdb
        # pdb.set_trace()
        #self.step_size=config.step_size

        # For regression
        self.lb = None
        self.ub = None


        # For auto label search.
        self.return_full_softmax = None
    
    def freeze_backbone(self, use_pe: bool=True):
        if use_pe:
            self.deberta = freezer.freeze_lm(self.deberta)
        else:
            self.deberta = freezer.unfreeze_lm(self.deberta)

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.deberta.device)
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


    def get_constrast_loss(self,
                    input_ids=None,
                    attention_mask=None,
                    mask_pos=None,
                    labels=None,
                    inputs_embeds=None):

        self.cos = nn.CosineSimilarity(dim=-1)


        _, sequence_mask_output_1 = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)
        _, sequence_mask_output_2 = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

        sequence_mask_output_1= self.lm_head.dense(sequence_mask_output_1)
        sequence_mask_output_2 = self.lm_head.dense(sequence_mask_output_2)
        # input_args = [input_ids, attention_mask, mask_pos, labels, None, 1]
        # embed = self.forward(*input_args)
        #
        # vat_args = [input_ids, attention_mask, mask_pos, labels, embed, 2]
        #
        # adv_logits, outputs = self.forward(*vat_args)
        #
        # logit_mask = F.softmax(logits, dim=-1)[torch.arange(adv_logits.size(0)), labels] > 0.7
        #
        # outputs = outputs[logit_mask]
        # seq_outputs = sequence_mask_output[logit_mask]
        # new_label = labels[logit_mask]
        # #
        # #
        # rand_perm = torch.randperm(outputs.size(0))
        # rand_outputs = outputs[rand_perm, :]
        # rand_label = new_label[rand_perm]
        # pair_label = (new_label == rand_label).long()
        #
        # seq_outputs = self.map(seq_outputs)
        # rand_outputs = self.map(rand_outputs)

        pair_labels = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # import  pdb
        # pdb.set_trace()

        contra_loss = self.contra_lc(sequence_mask_output_1.unsqueeze(1), sequence_mask_output_2.unsqueeze(0), pair_labels)

        if torch.isnan(contra_loss):
            return 0

        return contra_loss

    def embed_encode(self, input_ids):
        embedding_output = self.deberta.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, token_type_ids=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        batch_size = input_ids.size(0)

        # add prefix for prompt-tuning
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.deberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything

        outputs = self.deberta(
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

        #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

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

        if self.model_args.hybrid == 1:
            cls_logits = self.classifier(sequence_output)
            return (logits, cls_logits), sequence_mask_output

        return logits, sequence_mask_output


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            mask_pos=None,
            labels=None,
            inputs_embeds=None,
            fwd_type=0,
            block_flag=None,
            return_dict=None,
    ):
        
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                               mask_pos=mask_pos, inputs_embeds=inputs_embeds)

        elif fwd_type == 1:
            return self.embed_encode(input_ids)



        if (self.model_args.prompt_ptuning or self.model_args.prompt_prefix) and block_flag is not None:
            inputs_embeds = self.generate_continuous_prompt_inputs(input_ids, block_flag)

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, token_type_ids, mask_pos, inputs_embeds)

        if self.model_args.hybrid == 1:
            logits = logits[0]
            cls_logits = logits[1]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb),
                                      (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:

                if labels.shape == logits.shape:
                    loss = F.kl_div(F.log_softmax(logits, dim=-1, dtype=torch.float32),
                                    labels, reduction='batchmean')
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




class Debertav2ForPromptFinetuning(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)

        if self.config.use_pe:
            self.deberta = freezer.freeze_lm(self.deberta)
        self.cls = DebertaV2OnlyMLMHead(config)

        #self.deberta = DebertaModel(config)
        #self.cls = DebertaOnlyMLMHead(config)

        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = torch.nn.Linear(output_dim, self.num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.dropout = StableDropout(drop_out)

        classification_list = [self.pooler, self.dropout,self.classifier]

        self.classifier = nn.Sequential(*classification_list)
        # self.cls = DebertaV2OnlyMLMHead(config)

        self.map = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.K = 1
        self.step_size=1e-5
        # import pdb
        # pdb.set_trace()
        #self.step_size=config.step_size

        # For regression
        self.lb = None
        self.ub = None

        self.pre_seq_len = self.config.pre_seq_len
        # For auto label search.
        self.return_full_softmax = None
    
    def freeze_backbone(self, use_pe: bool=True):
        if use_pe:
            self.deberta = freezer.freeze_lm(self.deberta)
        else:
            self.deberta = freezer.unfreeze_lm(self.deberta)

    def embed_encode(self, input_ids):
        embedding_output = self.deberta.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        if inputs_embeds is None:
            outputs =  self.deberta(
                input_ids,
                attention_mask=attention_mask
            )
        else:
            outputs =  self.deberta(
                None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds
            )


        # Get <mask> token representation
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]


        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

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
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        fwd_type=0,
        block_flag=None,
        return_dict=None
    ):
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos, inputs_embeds=inputs_embeds)

        elif fwd_type == 1:
            return self.embed_encode(input_ids)

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

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
                                    labels, reduction='batchmean')
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                    if self.model_args.hybrid == 1:
                        cls_loss = loss_fct(cls_logits.view(-1, cls_logits.size(-1)), labels.view(-1))
                        loss = loss + cls_loss

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


class Debertav2PrefixForPromptFinetuning(DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)
        self.cls = DebertaV2OnlyMLMHead(config)

        #self.deberta = DebertaModel(config)
        #self.cls = DebertaOnlyMLMHead(config)

        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = torch.nn.Linear(output_dim, self.num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.dropout = StableDropout(drop_out)

        classification_list = [self.pooler, self.dropout,self.classifier]

        self.classifier = nn.Sequential(*classification_list)
        # self.cls = DebertaV2OnlyMLMHead(config)

        self.map = nn.Linear(config.hidden_size, config.hidden_size)
        self.init_weights()

        if self.config.use_pe:
            self.deberta = freezer.freeze_lm(self.deberta)

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None
        self.K = 1
        self.step_size=1e-5
        # import pdb
        # pdb.set_trace()
        #self.step_size=config.step_size

        # For regression
        self.lb = None
        self.ub = None


        # For auto label search.
        self.return_full_softmax = None
    
    def freeze_backbone(self, use_pe: bool=True):
        if use_pe:
            self.deberta = freezer.freeze_lm(self.deberta)
        else:
            self.deberta = freezer.unfreeze_lm(self.deberta)
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.deberta.device)
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
        embedding_output = self.deberta.embeddings.word_embeddings(input_ids)
        return embedding_output

    def encode(self, input_ids=None, attention_mask=None, mask_pos=None, inputs_embeds=None, return_full_softmax=False):
        batch_size = input_ids.size(0)

        # add prefix for prompt-tuning
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.deberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )


        # Get <mask> token representation
        sequence_output = outputs[0]
        # sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]


        # Logits over vocabulary tokens
        prediction_mask_scores = self.cls(sequence_mask_output)

        #sequence_mask_output = self.lm_head.dense(sequence_mask_output)

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
        mask_pos=None,
        labels=None,
        inputs_embeds=None,
        fwd_type=0,
        block_flag=None,
        return_dict=None,
    ):
        if fwd_type == 2:
            assert inputs_embeds is not None
            return self.encode(input_ids=input_ids, attention_mask=attention_mask, mask_pos=mask_pos, inputs_embeds=inputs_embeds)

        elif fwd_type == 1:
            return self.embed_encode(input_ids)

        logits, sequence_mask_output = self.encode(input_ids, attention_mask, mask_pos, inputs_embeds)

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
                                    labels, reduction='batchmean')
                else:
                    loss_fct = nn.CrossEntropyLoss()

                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                    if self.model_args.hybrid == 1:
                        cls_loss = loss_fct(cls_logits.view(-1, cls_logits.size(-1)), labels.view(-1))
                        loss = loss + cls_loss

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
