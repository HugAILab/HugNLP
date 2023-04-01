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
from processors.benchmark.glue.data_processor import GLUEProcessor, GLUEForInContextProcessor
# pre-training language model
from processors.pretraining.mlm.data_processor import MLMTextLineProcessor
# from processor.pretraining.mlm.data_processor import MLMGroupProcessor, MLMFromDisk, MLMLineByLineProcessor, WWMFromDisk
from processors.pretraining.kg_enhance_plm.data_process import WikiKPPLMSupervisedJsonProcessor
from processors.pretraining.causal_lm.data_processor import CausalLMITextLineProcessor, CausalLMInContextProcessor
# few-shot ner
from processors.ner.fewshot_ner.data_processor import SpanProtoFewNERDProcessor, SpanProtoCrossNERProcessor, TokenProtoFewNERDProcessor
# instruction-tuning
from processors.instruction_prompting.chinese_extractive_instruction.data_processor import ChineseExtractiveInstructionProcessor
from processors.instruction_prompting.incontext_learning.data_processor import CausalInContextClassificationProcessor
# code
from processors.code.code_clone.data_processor import CodeCloneProcessor
from processors.code.code_defect.data_processor import CodeDefectProcessor
# default applications
from processors.default_task_processors.data_processor import (
    DefaultSequenceClassificationProcessor,
    DefaultSequenceLabelingProcessor
)

# Pre-training Tasks
PRETRAINING_PROCESSORS = {
    "mlm_text_line": MLMTextLineProcessor,
    "causal_lm_text_line": CausalLMITextLineProcessor,
    "en_wiki_kpplm": WikiKPPLMSupervisedJsonProcessor,
}

# default task
DEFAULT_PROCESSORS = {
    "default_cls": DefaultSequenceClassificationProcessor,
    "default_labeling": DefaultSequenceLabelingProcessor
}

# Information Extraction Tasks
IE_PROCESSORS = {
    "sequence_proto": None,
    "span_proto_fewnerd": SpanProtoFewNERDProcessor,  # span-based proto for few-shot ner
    "token_proto_fewnerd": TokenProtoFewNERDProcessor,  # token-based proto for few-shot ner
}

# Benchmark
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
    "glue_instruction": GLUEForInContextProcessor, # instruction-tuning for glue
}

# Instruction / Prompting / In-context / Chain-of-Thought
INSTRUCTION_PROCESSORS = {
    "causal_instruction": None, # using causal instruction-tuning
    "zh_mrc_instruction": ChineseExtractiveInstructionProcessor, # using extractive-instruction for chinese
    "causal_incontext_cls": CausalInContextClassificationProcessor, # using causal in-context learning for cls tasks
    "causal_incontext": None, # using causal in-context
    "causal_chain_of_thought": None, # using causal chain-of-thought
}



CODE_PROCESSORS = {
    "code_clone": CodeCloneProcessor,
    "code_defect": CodeDefectProcessor,
    "code_refine": None,
    "code_translation": None,
    "code_summarization": None,

}

OTHER_PROCESSORS = {
    # pre-training language model
    # "mlm_from_disk": MLMFromDisk,
    # "wwm_from_disk": WWMFromDisk,
    # "mlm_line_by_line": MLMLineByLineProcessor,
    # "mlm_group": MLMGroupProcessor,
    # "causal_lm_incontext": CausalLMInContextProcessor,
    # "kgpretrain": PretrainWithKGFromDisk,
    # "kgpretrain_v2": KgV2Processor,

    # chinese instruction-tuning

    # "cpic": CPICProcessor,
}

PROCESSORS_LIST = [
    PRETRAINING_PROCESSORS,
    DEFAULT_PROCESSORS,
    IE_PROCESSORS,
    INSTRUCTION_PROCESSORS,
    BENCHMARKS_PROCESSORS,
    CODE_PROCESSORS,
    OTHER_PROCESSORS,
]

# PROCESSORS = dict(list(PRETRAINING_PROCESSORS.items()) + list(OTHER_PROCESSORS.items()))
PROCESSORS = dict()
for processor in PROCESSORS_LIST:
    PROCESSORS = dict(list(PROCESSORS.items()) + list(processor.items()))
