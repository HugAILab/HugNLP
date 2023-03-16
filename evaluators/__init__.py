# -*- coding: utf-8 -*-
# @Time    : 2023/3/15 3:35 下午
# @Author  : JianingWang
# @File    : __init__.py


# Models for pre-training
PRETRAIN_EVALUATOR_CLASSES = {
    "mlm": None,
    "auto_mlm": None,
    "causal_lm": None,
}


CLASSIFICATION_EVALUATOR_CLASSES = {
    "auto_cls": None, # huggingface cls
    "classification": None, # huggingface cls
    "head_cls": None, # use standard fine-tuning head for cls, e.g., bert+mlp
    "head_prefix_cls": None, # use standard fine-tuning head with prefix-tuning technique for cls, e.g., bert+mlp
    "head_ptuning_cls": None, # use standard fine-tuning head with p-tuning technique for cls, e.g., bert+mlp
    "head_adapter_cls": None, # use standard fine-tuning head with adapter-tuning technique for cls, e.g., bert+mlp
    "masked_prompt_cls": None, # use masked lm head technique for prompt-based cls, e.g., bert+mlm
    "masked_prompt_prefix_cls": None, # use masked lm head with prefix-tuning technique for prompt-based cls, e.g., bert+mlm
    "masked_prompt_ptuning_cls": None, # use masked lm head with p-tuning technique for prompt-based cls, e.g., bert+mlm
    "masked_prompt_adapter_cls": None, # use masked lm head with adapter-tuning technique for prompt-based cls, e.g., bert+mlm
    "causal_prompt_cls": None, # use causal lm head for prompt-tuning, e.g., gpt2+lm
}


TOKEN_CLASSIFICATION_EVALUATOR_CLASSES = {
    "ner": None,
}


SPAN_EXTRACTION_EVALUATOR_CLASSES = {
    "global_pointer": None,
}


FEWSHOT_EVALUATOR_CLASSES = {
    "sequence_proto": None,
    "span_proto": None,
    "token_proto": None,
}


CODE_EVALUATOR_CLASSES = {
    "code_cls": None,
    "code_generation": None,
}


# task_type 负责对应model类型
OTHER_EVALUATOR_CLASSES = {
    # sequence labeling
    "ner": None,
    "bert_softmax_ner": None,
    "roberta_softmax_ner": None,
    "albert_softmax_ner": None,
    "megatronbert_softmax_ner": None,
    "bert_crf_ner": None,
    "roberta_crf_ner": None,
    "albert_crf_ner": None,
    "megatronbert_crf_ner": None,
    "bert_span_ner": None,
    "roberta_span_ner": None,
    "albert_span_ner": None,
    "megatronbert_span_ner": None,
    # sequence matching
    "fusion_siamese": None,
    # multiple choice
    "multi_choice": None,
    "multi_choice_megatron": None,
    "multi_choice_megatron_rdrop": None,
    "megatron_multi_choice_tag": None,
    "roformer_multi_choice_tag": None,
    "multi_choice_tag": None,
    "duma": None,
    "duma_albert": None,
    "duma_megatron": None,
    # language modeling

    # "bert_mlm_acc": BertForMaskedLMWithACC,
    # "roformer_mlm_acc": RoFormerForMaskedLMWithACC,
    "bert_pretrain_kg": None,
    "bert_pretrain_kg_v2": None,
    "kpplm_roberta": None,
    "kpplm_deberta": None,

    # other
    "clue_wsc": None,
    "semeval7multitask": None,
}



# MODEL_CLASSES = dict(list(PRETRAIN_MODEL_CLASSES.items()) + list(OTHER_MODEL_CLASSES.items()))
EVALUATORS_LIST = [
    PRETRAIN_EVALUATOR_CLASSES,
    CLASSIFICATION_EVALUATOR_CLASSES,
    TOKEN_CLASSIFICATION_EVALUATOR_CLASSES,
    SPAN_EXTRACTION_EVALUATOR_CLASSES,
    FEWSHOT_EVALUATOR_CLASSES,
    CODE_EVALUATOR_CLASSES,
    OTHER_EVALUATOR_CLASSES
]


EVALUATORS_CLASSES = dict()
for evaluator_class in EVALUATORS_LIST:
    EVALUATORS_CLASSES = dict(list(EVALUATORS_CLASSES.items()) + list(evaluator_class.items()))
