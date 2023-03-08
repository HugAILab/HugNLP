# -*- coding: utf-8 -*-
# @Time    : 2021/12/6 3:35 下午
# @Author  : JianingWang
# @File    : __init__.py


# from models.chid_mlm import BertForChidMLM
from models.multiple_choice.duma import BertDUMAForMultipleChoice, AlbertDUMAForMultipleChoice, MegatronDumaForMultipleChoice
from models.span_extraction.global_pointer import BertForEffiGlobalPointer, RobertaForEffiGlobalPointer, RoformerForEffiGlobalPointer, MegatronForEffiGlobalPointer
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForMultipleChoice, BertTokenizer, \
    AutoModelForQuestionAnswering

from transformers.models.roformer import RoFormerTokenizer
from transformers.models.bert import BertTokenizerFast, BertForTokenClassification, BertTokenizer
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
# from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.bart.tokenization_bart import BartTokenizer
from transformers.models.t5.tokenization_t5 import T5Tokenizer


# from models.deberta import DebertaV2ForMultipleChoice, DebertaForMultipleChoice
# from models.fengshen.models.longformer import LongformerForMultipleChoice
from models.kg import BertForPretrainWithKG, BertForPretrainWithKGV2
from models.language_modeling.mlm import BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM, RoFormerForMaskedLM
# from models.sequence_classification.classification import build_cls_model
from models.multiple_choice.multiple_choice_tag import BertForTagMultipleChoice, RoFormerForTagMultipleChoice, MegatronBertForTagMultipleChoice
from models.multiple_choice.multiple_choice import MegatronBertForMultipleChoice, MegatronBertRDropForMultipleChoice
from models.semeval7 import DebertaV2ForSemEval7MultiTask
from models.sequence_matching.fusion_siamese import BertForFusionSiamese, BertForWSC
# from roformer import RoFormerForTokenClassification, RoFormerForSequenceClassification
from models.fewshot_learning.span_proto import SpanProto
from models.fewshot_learning.token_proto import TokenProto

from models.sequence_labeling.softmax_for_ner import BertSoftmaxForNer, RobertaSoftmaxForNer, AlbertSoftmaxForNer, MegatronBertSoftmaxForNer
from models.sequence_labeling.crf_for_ner import BertCrfForNer, RobertaCrfForNer, AlbertCrfForNer, MegatronBertCrfForNer
from models.span_extraction.span_for_ner import BertSpanForNer, RobertaSpanForNer, AlbertSpanForNer, MegatronBertSpanForNer

from models.language_modeling.kpplm import BertForWikiKGPLM, RoBertaKPPLMForProcessedWikiKGPLM, DeBertaKPPLMForProcessedWikiKGPLM
from models.language_modeling.causal_lm import GPT2ForCausalLM

from models.sequence_classification.head_cls import (
    BertForSequenceClassification, BertPrefixForSequenceClassification,
    BertPtuningForSequenceClassification, BertAdapterForSequenceClassification,
    RobertaForSequenceClassification, RobertaPrefixForSequenceClassification,
    RobertaPtuningForSequenceClassification,RobertaAdapterForSequenceClassification,
    BartForSequenceClassification, GPT2ForSequenceClassification
)

from models.sequence_classification.masked_prompt_cls import (
    PromptBertForSequenceClassification, PromptBertPtuningForSequenceClassification,
    PromptBertPrefixForSequenceClassification, PromptBertAdapterForSequenceClassification,
    PromptRobertaForSequenceClassification, PromptRobertaPtuningForSequenceClassification,
    PromptRobertaPrefixForSequenceClassification, PromptRobertaAdapterForSequenceClassification
)

from models.language_modeling.mlm import BertForMaskedLM

# Models for pre-training
PRETRAIN_MODEL_CLASSES = {
    "mlm": {
        "bert": BertForMaskedLM,
        "roberta": RobertaForMaskedLM,
        "albert": AlbertForMaskedLM,
        "roformer": RoFormerForMaskedLM,
    },
    "auto_mlm": AutoModelForMaskedLM,
    "causal_lm": {
        "gpt2": GPT2ForCausalLM,
        "bart": None,
        "t5": None,
    },
}

CLASSIFICATION_MODEL_CLASSES = {
    "auto_cls": AutoModelForSequenceClassification, # huggingface cls
    "classification": AutoModelForSequenceClassification, # huggingface cls
    "head_cls": {
        "bert": BertForSequenceClassification,
        "roberta": RobertaForSequenceClassification,
        "bart": BartForSequenceClassification,
        "gpt2": GPT2ForSequenceClassification
    }, # use standard fine-tuning head for cls, e.g., bert+mlp
    "head_prefix_cls": {
        "bert": BertPrefixForSequenceClassification,
        "roberta": RobertaPrefixForSequenceClassification,
    }, # use standard fine-tuning head with prefix-tuning technique for cls, e.g., bert+mlp
    "head_ptuning_cls": {
        "bert": BertPtuningForSequenceClassification,
        "roberta": RobertaPtuningForSequenceClassification,
    }, # use standard fine-tuning head with p-tuning technique for cls, e.g., bert+mlp
    "head_adapter_cls": {
        "bert": BertAdapterForSequenceClassification,
        "roberta": RobertaAdapterForSequenceClassification,
    }, # use standard fine-tuning head with adapter-tuning technique for cls, e.g., bert+mlp
    "masked_prompt_cls": {
        "bert": PromptBertForSequenceClassification,
        "roberta": PromptRobertaForSequenceClassification,
        # "deberta": PromptDebertaForSequenceClassification,
        # "deberta-v2": PromptDebertav2ForSequenceClassification,
    }, # use masked lm head technique for prompt-based cls, e.g., bert+mlm
    "masked_prompt_prefix_cls": {
        "bert": PromptBertPrefixForSequenceClassification,
        "roberta": PromptRobertaPrefixForSequenceClassification,
    #     "deberta": PromptDebertaPrefixForSequenceClassification,
    #     "deberta-v2": PromptDebertav2PrefixForSequenceClassification,
    }, # use masked lm head with prefix-tuning technique for prompt-based cls, e.g., bert+mlm
    "masked_prompt_ptuning_cls": {
        "bert": PromptBertPtuningForSequenceClassification,
        "roberta": PromptRobertaPtuningForSequenceClassification,
    #     "deberta": PromptDebertaPtuningForSequenceClassification,
    #     "deberta-v2": PromptDebertav2PtuningForSequenceClassification,
    }, # use masked lm head with p-tuning technique for prompt-based cls, e.g., bert+mlm
    "masked_prompt_adapter_cls": {
        "bert": PromptBertAdapterForSequenceClassification,
        "roberta": PromptRobertaAdapterForSequenceClassification,
    }, # use masked lm head with adapter-tuning technique for prompt-based cls, e.g., bert+mlm
    "causal_prompt_cls": {
        "gpt2": None,
        "bart": None,
        "t5": None,
    }, # use causal lm head for prompt-tuning, e.g., gpt2+lm
}

TOKEN_CLASSIFICATION_MODEL_CLASSES = {

}

SPAN_EXTRACTION_MODEL_CLASSES = {
    "global_pointer": {
        "bert": BertForEffiGlobalPointer,
        "roberta": RobertaForEffiGlobalPointer,
        "roformer": RoformerForEffiGlobalPointer,
        "megatronbert": MegatronForEffiGlobalPointer
    },
}


FEWSHOT_MODEL_CLASSES = {
    "sequence_proto": None,
    "span_proto": SpanProto,
    "token_proto": TokenProto,
}


CODE_MODEL_CLASSES = {
    "ner": AutoModelForTokenClassification,
}

# task_type 负责对应model类型
OTHER_MODEL_CLASSES = {
    # sequence labeling
    "ner": AutoModelForTokenClassification,
    "bert_softmax_ner": BertSoftmaxForNer,
    "roberta_softmax_ner": RobertaSoftmaxForNer,
    "albert_softmax_ner": AlbertSoftmaxForNer,
    "megatronbert_softmax_ner": MegatronBertSoftmaxForNer,
    "bert_crf_ner": BertCrfForNer,
    "roberta_crf_ner": RobertaCrfForNer,
    "albert_crf_ner": AlbertCrfForNer,
    "megatronbert_crf_ner": MegatronBertCrfForNer,
    "bert_span_ner": BertSpanForNer,
    "roberta_span_ner": RobertaSpanForNer,
    "albert_span_ner": AlbertSpanForNer,
    "megatronbert_span_ner": MegatronBertSpanForNer,
    # sequence matching
    "fusion_siamese": BertForFusionSiamese,
    # multiple choice
    "multi_choice": AutoModelForMultipleChoice,
    "multi_choice_megatron": MegatronBertForMultipleChoice,
    "multi_choice_megatron_rdrop": MegatronBertRDropForMultipleChoice,
    "megatron_multi_choice_tag": MegatronBertForTagMultipleChoice,
    "roformer_multi_choice_tag": RoFormerForTagMultipleChoice,
    "multi_choice_tag": BertForTagMultipleChoice,
    "duma": BertDUMAForMultipleChoice,
    "duma_albert": AlbertDUMAForMultipleChoice,
    "duma_megatron": MegatronDumaForMultipleChoice,
    # language modeling

    # "bert_mlm_acc": BertForMaskedLMWithACC,
    # "roformer_mlm_acc": RoFormerForMaskedLMWithACC,
    "bert_pretrain_kg": BertForPretrainWithKG,
    "bert_pretrain_kg_v2": BertForPretrainWithKGV2,
    "kpplm_roberta": RoBertaKPPLMForProcessedWikiKGPLM,
    "kpplm_deberta": DeBertaKPPLMForProcessedWikiKGPLM,

    # other
    "clue_wsc": BertForWSC,
    "semeval7multitask": DebertaV2ForSemEval7MultiTask,
    # "debertav2_multi_choice": DebertaV2ForMultipleChoice,
    # "deberta_multi_choice": DebertaForMultipleChoice,
    # "qa": AutoModelForQuestionAnswering,
    # "roformer_cls": RoFormerForSequenceClassification,
    # "roformer_ner": RoFormerForTokenClassification,
    # "fensheng_multi_choice": LongformerForMultipleChoice,
    # "chid_mlm": BertForChidMLM,
}

# MODEL_CLASSES = dict(list(PRETRAIN_MODEL_CLASSES.items()) + list(OTHER_MODEL_CLASSES.items()))
MODEL_CLASSES_LIST = [
    PRETRAIN_MODEL_CLASSES,
    CLASSIFICATION_MODEL_CLASSES,
    SPAN_EXTRACTION_MODEL_CLASSES,
    FEWSHOT_MODEL_CLASSES,
    CODE_MODEL_CLASSES,
    OTHER_MODEL_CLASSES,
]


MODEL_CLASSES = dict()
for model_class in MODEL_CLASSES_LIST:
    MODEL_CLASSES = dict(list(MODEL_CLASSES.items()) + list(model_class.items()))

# model_type 负责对应tokenizer
TOKENIZER_CLASSES = {
    "bert": BertTokenizerFast,
    "roberta": RobertaTokenizer,
    "wobert": RoFormerTokenizer,
    "roformer": RoFormerTokenizer,
    "bigbird": BertTokenizerFast,
    "erlangshen": BertTokenizerFast,
    "deberta": BertTokenizer,
    "roformer_v2": BertTokenizerFast,
    "gpt2": GPT2TokenizerFast,
    "megatronbert": BertTokenizerFast,
    "bart": BartTokenizer,
    "t5": T5Tokenizer,
}
