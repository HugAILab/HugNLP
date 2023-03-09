# -*- coding: utf-8 -*-
# @Time    : 2021/12/2 4:49 下午
# @Author  : JianingWang
# @File    : __init__.py

# benchmarks
from processors.benchmark.cluemrc.chid_mlm import ChidMLMProcessor
from processors.benchmark.cluemrc.c3 import C3Processor
from processors.benchmark.cluemrc.chid import ChidTagProcessor
from processors.benchmark.cluemrc.cmrc2018 import CMRCProcessor, CMRCForGlobalPointerProcessor
from processors.benchmark.cluemrc.data_processor import CLUEMRCProcessor
from processors.benchmark.clue.data_processor import CLUEProcessor, TnewsEFLProcessor, CSLEFLProcessor
from processors.benchmark.cluener.data_processor import CLUENERProcessor
from processors.benchmark.fewclue.data_processor import InstructionMRCForFewCLUEProcessor
from processors.benchmark.glue.data_processor import GLUEProcessor
# pre-training language model
from processors.pretraining.mlm.data_processor import MLMTextLineProcessor
# from processor.pretraining.mlm.data_processor import MLMGroupProcessor, MLMFromDisk, MLMLineByLineProcessor, WWMFromDisk
from processors.pretraining.kg_enhance_plm.data_process import WikiKPPLMSupervisedJsonProcessor
from processors.pretraining.causal_lm.data_processor import CausalLMITextLineProcessor, CausalLMInContextProcessor
# few-shot ner
from processors.ner.fewshot_ner.data_processor import SpanProtoFewNERDProcessor, SpanProtoCrossNERProcessor, TokenProtoFewNERDProcessor
# chinese instruction-tuning
from processors.instruction_prompting.chinese_extractive_instruction.data_processor import ChineseExtractiveInstructionProcessor

# default applications
from processors.default_task_processors.data_processor import DefaultSequenceClassificationProcessor

# Pre-training Tasks
PRETRAINING_PROCESSORS = {
    "mlm_text_line": MLMTextLineProcessor,
    "causal_lm_text_line": CausalLMITextLineProcessor,
    "en_wiki_kpplm": WikiKPPLMSupervisedJsonProcessor,
}

# Information Extraction Tasks
IE_PROCESSORS = {
    "span_proto_fewnerd": SpanProtoFewNERDProcessor,  # span-based proto for few-shot ner
    "token_proto_fewnerd": TokenProtoFewNERDProcessor,  # token-based proto for few-shot ner
}

BENCHMARKS_PROCESSORS = {
    "clue": CLUEProcessor,
    "clue_ner": CLUENERProcessor,
    "clue_tnews_efl": TnewsEFLProcessor,  # clue Tnews改造
    "clue_csl_efl": CSLEFLProcessor,  # clue Csl改造
    "c3": C3Processor,
    "clue_chid": ChidTagProcessor,  # clue的chid任务
    "cmrc": CMRCProcessor,  # clue的cmrc任务
    "chid_mlm": ChidMLMProcessor,
    "clue_mrc_style": CLUEMRCProcessor,  # clue任务转换为mrc模式
    "cmrc18_global_pointer": CMRCForGlobalPointerProcessor,
    "fewclue_instruction": InstructionMRCForFewCLUEProcessor,
    "glue": GLUEProcessor,  # glue
}

INSTRUCTION_PROCESSORS = {
    "zh_mrc_instruction": ChineseExtractiveInstructionProcessor, # 使用mrc
}



CODE_PROCESSORS = {
    "code_clone": None,
    "code_defect": None,
    "code_refine": None,
    "code_translation": None,
    "code_summarization": None,

}

OTHER_PROCESSORS = {
    "default_cls": DefaultSequenceClassificationProcessor,
    # pre-training language model
    # "mlm_from_disk": MLMFromDisk,
    # "wwm_from_disk": WWMFromDisk,
    # "mlm_line_by_line": MLMLineByLineProcessor,
    # "mlm_group": MLMGroupProcessor,
    "causal_lm_incontext": CausalLMInContextProcessor,
    # "kgpretrain": PretrainWithKGFromDisk,
    # "kgpretrain_v2": KgV2Processor,

    # chinese instruction-tuning

    # "cpic": CPICProcessor,
}

PROCESSORS_LIST = [
    PRETRAINING_PROCESSORS,
    IE_PROCESSORS,
    INSTRUCTION_PROCESSORS,
    BENCHMARKS_PROCESSORS,
    OTHER_PROCESSORS,
]

# PROCESSORS = dict(list(PRETRAINING_PROCESSORS.items()) + list(OTHER_PROCESSORS.items()))
PROCESSORS = dict()
for processor in PROCESSORS_LIST:
    PROCESSORS = dict(list(PROCESSORS.items()) + list(processor.items()))
