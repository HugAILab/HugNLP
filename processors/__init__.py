# -*- coding: utf-8 -*-
# @Time    : 2021/12/2 4:49 下午
# @Author  : JianingWang
# @File    : __init__.py

# benchmark
from processors.benchmark.cluemrc.chid_mlm import ChidMLMProcessor
from processors.benchmark.cluemrc.c3 import C3Processor
from processors.benchmark.cluemrc.chid import ChidTagProcessor
from processors.benchmark.cluemrc.cmrc2018 import CMRCProcessor, CMRCGPProcessor
from processors.benchmark.cluemrc.data_processor import CLUEMRCProcessor
from processors.benchmark.clue.data_processor import CLUEProcessor, TnewsEFLProcessor, CSLEFLProcessor
from processors.benchmark.cluener.data_processor import CLUENERProcessor
from processors.benchmark.fewclue.data_processor import InstructionMRCForFewCLUEProcessor
# pre-training language model
from processors.pretraining.mlm.data_processor import MLMTextLineProcessor
# from processor.pretraining.mlm.data_processor import MLMGroupProcessor, MLMFromDisk, MLMLineByLineProcessor, WWMFromDisk
from processors.pretraining.kg_enhance_plm.data_process import WikiKPPLMSupervisedJsonProcessor
from processors.pretraining.causal_lm.data_processor import CausalLMITextLineProcessor
# few-shot ner
from processors.ner.fewshot_ner.data_processor import SpanProtoFewNERDProcessor, SpanProtoCrossNERProcessor, TokenProtoFewNERDProcessor
# seq_ner
from processors.ner.seq_ner.data_process import Conll2003Processor
# chinese instruction-tuning
from processors.chinese_instruction_tuning.data_process import ChineseInstructionMRCProcessor

# Pre-training Tasks
PRETRAINING_PROCESSORS = {
    "mlm_text_line": MLMTextLineProcessor, 
    "causal_lm_text_line": CausalLMITextLineProcessor,
}


# Information Extraction Tasks
IE_PROCESSORS = {
    "span_proto_fewnerd": SpanProtoFewNERDProcessor, # span-based proto for few-shot ner
    "token_proto_fewnerd": TokenProtoFewNERDProcessor, # token-based proto for few-shot ner
}

OTHER_PROCESSORS = {
    # benchmark
    'clue': CLUEProcessor,
    'clue_ner': CLUENERProcessor,
    'clue_tnews_efl': TnewsEFLProcessor, # clue Tnews改造
    'clue_csl_efl': CSLEFLProcessor, # clue Csl改造
    'c3': C3Processor,
    'clue_chid': ChidTagProcessor, # clue的chid任务
    'cmrc': CMRCProcessor, # clue的cmrc任务
    'chid_mlm': ChidMLMProcessor,
    'clue_mrc_style': CLUEMRCProcessor, # clue任务转换为mrc模式
    'cmrc_gp': CMRCGPProcessor,
    'fewclue_instruction': InstructionMRCForFewCLUEProcessor,
    # pre-training language model
    # 'mlm_from_disk': MLMFromDisk,
    # 'wwm_from_disk': WWMFromDisk,
    # 'mlm_line_by_line': MLMLineByLineProcessor,
    # 'mlm_group': MLMGroupProcessor,
    # 'causal_lm_incontext': CausalLMInContextProcessor,
    # 'kgpretrain': PretrainWithKGFromDisk,
    # 'kgpretrain_v2': KgV2Processor,
    'en_wiki_kpplm': WikiKPPLMSupervisedJsonProcessor,
    # ner
    'conll2003': Conll2003Processor,
    # chinese instruction-tuning
    'zh_mrc_instruction': ChineseInstructionMRCProcessor,
    # 'cpic': CPICProcessor,
    
}

PROCESSORS_LIST = [
    PRETRAINING_PROCESSORS, 
    IE_PROCESSORS,
    OTHER_PROCESSORS,
]


# PROCESSORS = dict(list(PRETRAINING_PROCESSORS.items()) + list(OTHER_PROCESSORS.items()))
PROCESSORS = dict()
for processor in PROCESSORS_LIST:
    PROCESSORS = dict(list(PROCESSORS.items()) + list(processor.items()))