# Pre-built Applications Overview

We demonstrate all pre-built applications in HugNLP.

| **Applications** | **Runing Tasks** | **Task Notes** | **PLM Models** | **Documents** |
| --- | --- | --- | --- | --- |
| **Default Application** | run_seq_cls.sh | **Goal**: Standard **Fine-tuning** or **Prompt-tuning** for sequence classification on user-defined dataset. <br> **Path**: applications/default_applications | BERT, RoBERTa, DeBERTa | [click](./default_tasks/default_sequence_classification.md) |
|  | run_seq_labeling.sh | **Goal**: Standard **Fine-tuning** for sequence labeling on user-defined dataset. <br> **Path**: applications/default_applications | BERT, RoBERTa, ALBERT |   |
| **Pre-training** | run_pretrain_mlm.sh | **Goal**: Pre-training via **Masked Language Modeling** (MLM). <br> **Path**: applications/pretraining/ | BERT, RoBERTa | [click](./pretraining/Masked%20LM%20for%20Continual%20Pre-training.md) |
|  | run_pretrain_casual_lm.sh | **Goal**: Pre-training via **Causal Language Modeling** (CLM). <br> **Path**: applications/pretraining | BERT, RoBERTa | [click](./pretraining/Causal%20LM%20for%20Continual%20Pre-training.md) |
| **GLUE Benchmark** | run_glue.sh | **Goal**: Standard **Fine-tuning** or **Prompt-tuning** for GLUE classification tasks. <br> **Path**: applications/benchmark/glue | BERT, RoBERTa, DeBERTa |  |
|  | run_causal_incontext_glue.sh | **Goal**: **In-context learning** for GLUE classification tasks. <br> **Path**: applications/benchmark/glue | GPT-2 |  |
| **CLUE Benchmark** | clue_finetune_dev.sh | **Goal**: Standard **Fine-tuning** and **Prompt-tuning** for CLUE classification task。 <br> **Path**: applications/benchmark/clue | BERT, RoBERTa, DeBERTa |  |
|  | run_clue_cmrc.sh | **Goal**: Standard **Fine-tuning** for CLUE CMRC2018 task. <br> **Path**: applications/benchmark/cluemrc | BERT, RoBERTa, DeBERTa |  |
|  | run_clue_c3.sh | **Goal**: Standard **Fine-tuning** for CLUE C3 task. <br> **Path**: applications/benchmark/cluemrc | BERT, RoBERTa, DeBERTa |  |
|  | run_clue_chid.sh | **Goal**: Standard **Fine-tuning** for CLUE CHID task. <br> **Path**: applications/benchmark/cluemrc | BERT, RoBERTa, DeBERTa |  |
| **Instruction-Prompting** | run_causal_instruction.sh | **Goal**: **Cross-task training** via generative Instruction-tuning based on causal PLM. <font color='red'>**You can use it to train a small ChatGPT**</font>. <br> **Path**: applications/instruction_prompting/instruction_tuning | GPT2 | [click](./instruction_prompting/generative_instruction_tuning.md) |
|  | run_zh_extract_instruction.sh | **Goal**: **Cross-task training** via extractive Instruction-tuning based on Global Pointer model. <br> **Path**: applications/instruction_prompting/chinese_instruction | BERT, RoBERTa, DeBERTa | [click](./instruction_prompting/extractive_instruction_tuning.md) |
|  | run_causal_incontext_cls.sh | **Goal**: **In-context learning** for user-defined classification tasks. <br> **Path**: applications/instruction_prompting/incontext_learning | GPT-2 | [click](./instruction_prompting/incontext_learning_for_cls.md) |
| **Information Extraction** | run_extractive_unified_ie.sh | **Goal**: **HugIE**: training a unified chinese information extraction via extractive instruction-tuning. <br> **Path**: applications/information_extraction/HugIE | BERT, RoBERTa, DeBERTa | [click](./information_extraction/HugIE.md) |
|  | api_test.py | **Goal**: HugIE: API test. <br> **Path**: applications/information_extraction/HugIE | - | [click](./information_extraction/HugIE.md) |
|  | run_fewnerd.sh | **Goal**: **Prototypical learning** for named entity recognition, including SpanProto, TokenProto <br> **Path**: applications/information_extraction/fewshot_ner | BERT |  |
| **Code NLU** | run_clone_cls.sh | **Goal**: Standard **Fine-tuning** for code clone classification task. <br> **Path**: applications/code/code_clone | CodeBERT, CodeT5, GraphCodeBERT, PLBART |  |
|  | run_defect_cls.sh | **Goal**: Standard **Fine-tuning** for code defect classification task. <br> **Path**: applications/code/code_defect | CodeBERT, CodeT5, GraphCodeBERT, PLBART |  |




# Pre-built Application Settings

We show the settings that matched with each pre-built application.

Notes:
- ✅: Have finished
- ⌛️: To do
- ⛔️: Not-available

| **Applications** | **Runing Tasks** | **Adv-training** | **Parameter-efficient** | **Pattern-Verbalizer** | **Instruction-Prompting** | **Self-training** | **Calibration** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Default Application** | run_seq_cls.sh | ✅ | ✅ | ✅ |   |   |   |
|  | run_seq_labeling.sh | ✅ | ✅ | ✅ |   |   |   |
| **Pre-training** | run_pretrain_mlm.sh | ✅ |  |  |  |   |
|  | run_pretrain_casual_lm.sh | ✅ |  |  |  |  |   |   |
| **GLUE Benchmark** | run_glue.sh | ✅ | ✅ | ✅ |  |  |   |   |
|  | run_causal_incontext_glue.sh | ✅ | ✅ | ✅ | ✅ |  |  ✅ |
| **CLUE Benchmark** | clue_finetune_dev.sh | ✅ | ✅ | ✅ |  |  |   |   |
|  | run_clue_cmrc.sh | ✅ |  |  |  |  |   |   |
|  | run_clue_c3.sh | ✅ |  |  |  |  |   |   |
|  | run_clue_chid.sh | ✅ |  |  |  |  |   |   |
| **Instruction-Prompting** | run_causal_instruction.sh | ✅ |  |  | ✅ |  |   |   |
|  | run_zh_extract_instruction.sh | ✅ |  | ✅ | ✅ |  |   |   |
|  | run_causal_incontext_cls.sh | ⛔️ | ⛔️ | ✅ | ✅ |   | ✅ |   |
| **Information Extraction** | run_extractive_unified_ie.sh | ✅ |  |  |  |  |   |   |
|  | api_test.py | ⛔️ |  |  |  |  |   |   |
|  | run_fewnerd.sh | ✅ |  |  |  |  |   |   |
| **Code NLU** | run_clone_cls.sh | ✅ | ✅ |  |  |  |   |   |
|  | run_defect_cls.sh | ✅ | ✅ |  |  |  |   |   |
