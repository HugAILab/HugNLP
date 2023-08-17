# -*- coding: utf-8 -*-
# @Time    : 2023/3/15 3:35 下午
# @Author  : JianingWang
# @File    : __init__.py

from evaluators.language_modeling_evaluator import MaskedLanguageModelingEvaluator, CausalLanguageModelingEvaluator
from evaluators.sequence_classification_evaluator import SequenceClassificationEvaluator, CausalSequenceClassificationEvaluator
from evaluators.token_classification_evaluator import TokenClassificationEvaluator
from evaluators.span_extraction_evaluator import SpanExtractionEvaluator
from evaluators.multi_choice_evaluator import MultiChoiceEvaluator
from evaluators.reinforcement_learning_evaluator import PairwiseRewardEvaluator

# Models for pre-training
PRETRAIN_EVALUATOR_CLASSES = {
    "mlm": MaskedLanguageModelingEvaluator,
    "auto_mlm": MaskedLanguageModelingEvaluator,
    "causal_lm": CausalLanguageModelingEvaluator,
    "auto_causal_lm": CausalLanguageModelingEvaluator,
}


CLASSIFICATION_EVALUATOR_CLASSES = {
    "auto_cls": SequenceClassificationEvaluator, # huggingface cls
    "classification": SequenceClassificationEvaluator, # huggingface cls
    "head_cls": SequenceClassificationEvaluator, # use standard fine-tuning head for cls, e.g., bert+mlp
    "head_prefix_cls": SequenceClassificationEvaluator, # use standard fine-tuning head with prefix-tuning technique for cls, e.g., bert+mlp
    "head_ptuning_cls": SequenceClassificationEvaluator, # use standard fine-tuning head with p-tuning technique for cls, e.g., bert+mlp
    "head_adapter_cls": SequenceClassificationEvaluator, # use standard fine-tuning head with adapter-tuning technique for cls, e.g., bert+mlp
    "masked_prompt_cls": SequenceClassificationEvaluator, # use masked lm head technique for prompt-based cls, e.g., bert+mlm
    "masked_prompt_prefix_cls": SequenceClassificationEvaluator, # use masked lm head with prefix-tuning technique for prompt-based cls, e.g., bert+mlm
    "masked_prompt_ptuning_cls": SequenceClassificationEvaluator, # use masked lm head with p-tuning technique for prompt-based cls, e.g., bert+mlm
    "masked_prompt_adapter_cls": SequenceClassificationEvaluator, # use masked lm head with adapter-tuning technique for prompt-based cls, e.g., bert+mlm
    "causal_prompt_cls": CausalSequenceClassificationEvaluator, # use causal lm head for prompt-tuning, e.g., gpt2+lm
}


TOKEN_CLASSIFICATION_EVALUATOR_CLASSES = {
    "auto_token_cls": TokenClassificationEvaluator,
    "head_softmax_token_cls": TokenClassificationEvaluator,
    "head_crf_token_cls": TokenClassificationEvaluator,
}


SPAN_EXTRACTION_EVALUATOR_CLASSES = {
    "global_pointer": SpanExtractionEvaluator,
}


FEWSHOT_EVALUATOR_CLASSES = {
    "sequence_proto": None,
    "span_proto": SpanExtractionEvaluator,
    "token_proto": TokenClassificationEvaluator,
}


CODE_EVALUATOR_CLASSES = {
    "code_cls": SequenceClassificationEvaluator,
    "code_generation": None,
}

REINFORCEMENT_MODEL_CLASSES = {
    "causal_actor": None,
    "auto_critic": None, 
    "rl_reward": PairwiseRewardEvaluator,
}

# task_type 负责对应model类型
OTHER_EVALUATOR_CLASSES = {
    # sequence labeling
    "bert_span_ner": SpanExtractionEvaluator,
    "roberta_span_ner": SpanExtractionEvaluator,
    "albert_span_ner": SpanExtractionEvaluator,
    "megatronbert_span_ner": SpanExtractionEvaluator,
    # sequence matching
    "fusion_siamese": SequenceClassificationEvaluator,
    # multiple choice
    "multi_choice": MultiChoiceEvaluator,
    "multi_choice_megatron": MultiChoiceEvaluator,
    "multi_choice_megatron_rdrop": MultiChoiceEvaluator,
    "megatron_multi_choice_tag": MultiChoiceEvaluator,
    "roformer_multi_choice_tag": MultiChoiceEvaluator,
    "multi_choice_tag": MultiChoiceEvaluator,
    "duma": MultiChoiceEvaluator,
    "duma_albert": MultiChoiceEvaluator,
    "duma_megatron": MultiChoiceEvaluator,
    # language modeling

    # "bert_mlm_acc": BertForMaskedLMWithACC,
    # "roformer_mlm_acc": RoFormerForMaskedLMWithACC,
    "bert_pretrain_kg": MaskedLanguageModelingEvaluator,
    "bert_pretrain_kg_v2": MaskedLanguageModelingEvaluator,
    "kpplm_roberta": MaskedLanguageModelingEvaluator,
    "kpplm_deberta": MaskedLanguageModelingEvaluator,

    # other
    "clue_wsc": SequenceClassificationEvaluator,
    "semeval7multitask": SequenceClassificationEvaluator,
}



# MODEL_CLASSES = dict(list(PRETRAIN_MODEL_CLASSES.items()) + list(OTHER_MODEL_CLASSES.items()))
EVALUATORS_LIST = [
    PRETRAIN_EVALUATOR_CLASSES,
    CLASSIFICATION_EVALUATOR_CLASSES,
    TOKEN_CLASSIFICATION_EVALUATOR_CLASSES,
    SPAN_EXTRACTION_EVALUATOR_CLASSES,
    FEWSHOT_EVALUATOR_CLASSES,
    CODE_EVALUATOR_CLASSES,
    REINFORCEMENT_MODEL_CLASSES,
    OTHER_EVALUATOR_CLASSES
]


EVALUATORS_CLASSES = dict()
for evaluator_class in EVALUATORS_LIST:
    EVALUATORS_CLASSES = dict(list(EVALUATORS_CLASSES.items()) + list(evaluator_class.items()))
