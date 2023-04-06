
<p align="center">
    <br>
    <img src="images/logo.png" width="360"/>
    <br>
</p>

<p align="center" style="font-size:22px;"> <b> Welcome to use HugNLP. 🤗 Hugging for NLP! </b>
</p>

<div align="center">

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/wjn1996/HugNLP/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/wjn1996/HugNLP/tree/main)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/wjn1996/HugNLP.svg)](https://github.com/wjn1996/HugNLP/pull/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![arXiv](https://img.shields.io/badge/arXiv-2302.14286-b31b1b.svg)](https://arxiv.org/abs/2302.14286)

</div>

# About HugNLP

HugNLP is a novel development and application library based on [Hugging Face](https://huggingface.co/) for improving the convenience and effectiveness of NLP researchers. The founder and main developer is [Jianing Wang](https://wjn1996.github.io/). The collaborators are [Nuo Chen](https://github.com/nchen909) and [Qiushi Sun](https://github.com/QiushiSun).

## **News & Highlights

- 🆕 [23-04-06]: Develop a samll ChatGPT, naming HugChat! You can chat with HugNLP! [[see doc](./documents/instruction_prompting/generative_instruction_tuning.md)]
- 🆕 [23-04-02]: Add GPT-style instruction-tuning. You can continual train a small-scale ChatGPT! [[see doc](./documents/instruction_prompting/generative_instruction_tuning.md)]
- 🆕 [23-03-21]: Finish GPT-style in-context learning for sequence classification. [[see doc](./documents/instruction_prompting/incontext_learning_for_cls.md)]
- 🆕 [23-03-13]: Add code clone detection and defect task. You can train clone and defect for user-defined dataset.
- 🆕 [23-03-03]: Add HugIE API and corresponding training script. You can use it to perform information extraction on Chinese data. [[see doc](./documents/information_extraction/HugIE.md)]
- 🆕 [23-02-18]: The HugNLP is open!

# Architecture

The framework overview is shown as follows:

<p align="center">
    <br>
    <img src="images/overview.png" width="80%"/>
    <br>
<p>

### Models

In HugNLP, we provide some popular transformer-based models as backbones, such as BERT, RoBERTa, GPT-2, etc. We also release our pre-built KP-PLM, a novel knowledge-enhanced pre-training paradigm to inject factual knowledge and can be easily used for arbitrary PLMs.
Apart from basic PLMs, we also implement some task-specific models, involving sequence classification, matching, labeling, span extraction, multi-choice, and text generation.
Notably, we develop standard fine-tuning (based on CLS Head and prompt-tuning models that enable PLM tuning on classification tasks.
For few-shot learning settings, HugNLP provides a prototypical network in both few-shot text classification and named entity recognition (NER).

In addition, we also incorporate some plug-and-play utils in HugNLP.

1. Parameter Freezing. If we want to perform parameter-efficient learning, which aims to freeze some parameters in PLMs to improve the training efficiency, we can set the configure `use_freezing` and freeze the backbone. A use case is shown in Code.
2. Uncertainty Estimation aims to calculate the model certainty when in semi-supervised learning.
3. We also design Prediction Calibration, which can be used to further improve the accuracy by calibrating the distribution and alleviating the semantics bias problem.

### Processors

Processors aim to load the dataset and process the task examples in a pipeline containing sentence tokenization, sampling, and tensor generation.
Specifically, users can directly obtain the data through `load_dataset`, which can directly download it from the Internet or load it from the local disk.
For different tasks, users should define a task-specific data collator, which aims to transform the original examples into model input tensor features.

### Applications

It provides rich modules for users to build real-world applications and products by selecting among an array of settings from Models and Processors.

# Core Capacities

We provide some core capacities to support the NLP downstream applications.

### Knowledge-enhanced Pre-trained Language Model

Conventional pre-training methods lack factual knowledge.
To deal with this issue, we present KP-PLM with a novel knowledge prompting paradigm for knowledge-enhanced pre-training.
Specifically, we construct a knowledge sub-graph for each input text by recognizing entities and aligning with the knowledge base and decompose this sub-graph into multiple relation paths, which can be directly transformed into language prompts.

### Prompt-based Fine-tuning

Prompt-based fine-tuning aims to reuse the pre-training objective (e.g., Masked Language Modeling, Causal Language Modeling) and utilizes a well-designed template and verbalizer to make predictions, which has achieved great success in low-resource settings.
We integrate some novel approaches into HugNLP, such as PET, P-tuning, etc.

### Instruction Tuning & In-Context Learning

Instruction-tuning and in-context learning enable few/zero-shot learning without parameter update, which aims to concatenate the task-aware instructions or example-based demonstrations to prompt GPT-style PLMs to generate reliable responses.
So, all the NLP tasks can be unified into the same format and can substantially improve the models" generalization.
Inspired by this idea, we extend it into other two paradigms:

1. extractive-style paradigm: we unify various NLP tasks into span extraction, which is the same as extractive question answering.
2. inference-style paradigm: all the tasks can be viewed as natural language inference to match the relations between inputs and outputs.
3. generative-style paradigm: we unify all the tasks into generative format, and train the causal models based on instruction-tuning, in-context learning or chain-of-thought.

### Self-training with Uncertainty Estimation

Self-training can address the labeled data scarcity issue by leveraging the large-scale unlabeled data in addition to labeled data, which is one of the mature paradigms in semi-supervised learning.
However, the standard self-training may generate too much noise, inevitably degrading the model performance due to confirmation bias.
Thus, we present uncertainty-aware self-training. Specifically, we train a teacher model on few-shot labeled data, and then use Monte Carlo (MC) dropout technique in Bayesian neural network (BNN) to approximate the model certainty, and judiciously select the examples that have a higher model certainty of the teacher.

### Parameter-Efficient Learning

To improve the training efficiency of HugNLP, we also implement parameter-efficient learning, which aims to freeze some parameters in the backbone so that we only tune a few parameters during model training.
We develop some novel parameter-efficient learning approaches, such as Prefix-tuning, Adapter-tuning, BitFit and LoRA, etc.

# Installation

> git clone https://github.com/wjn1996/HugNLP.git
>
> cd HugNLP
>
> python3 setup.py install

At present, the project is still being developed and improved, and there may be some `bugs` in use, please understand. We also look forward to your being able to ask issues or committing some valuable pull requests.

# Pre-built Applications Overview

We demonstrate all pre-built applications in HugNLP. You can choose one application to use HugNLP. You can also click the link to see the details document.

| **Applications**           | **Runing Tasks**        | **Task Notes**                                                                                                                                                                                                                                                    | **PLM Models**                    | **Documents**                                                           |
| -------------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------- |
| **Default Application**    | run_seq_cls.sh                | **Goal**: Standard **Fine-tuning** or **Prompt-tuning** for sequence classification on user-defined dataset. `<br>` **Path**: applications/default_applications                                                                               | BERT, RoBERTa, DeBERTa                  | [click](./documents/default_tasks/default_sequence_classification.md)            |
|                                  | run_seq_labeling.sh           | **Goal**: Standard **Fine-tuning** for sequence labeling on user-defined dataset. `<br>` **Path**: applications/default_applications                                                                                                                | BERT, RoBERTa, ALBERT                   |                                                                               |
| **Pre-training**           | run_pretrain_mlm.sh           | **Goal**: Pre-training via **Masked Language Modeling** (MLM). `<br>` **Path**: applications/pretraining/                                                                                                                                           | BERT, RoBERTa                           | [click](./documents/pretraining/Masked%20LM%20for%20Continual%20Pre-training.md) |
|                                  | run_pretrain_casual_lm.sh     | **Goal**: Pre-training via **Causal Language Modeling** (CLM). `<br>` **Path**: applications/pretraining                                                                                                                                            | BERT, RoBERTa                           | [click](./documents/pretraining/Causal%20LM%20for%20Continual%20Pre-training.md) |
| **GLUE Benchmark**         | run_glue.sh                   | **Goal**: Standard **Fine-tuning** or **Prompt-tuning** for GLUE classification tasks. `<br>` **Path**: applications/benchmark/glue                                                                                                           | BERT, RoBERTa, DeBERTa                  |                                                                               |
|                                  | run_causal_incontext_glue.sh  | **Goal**: **In-context learning** for GLUE classification tasks. `<br>` **Path**: applications/benchmark/glue                                                                                                                                       | GPT-2                                   |                                                                               |
| **CLUE Benchmark**         | clue_finetune_dev.sh          | **Goal**: Standard **Fine-tuning** and **Prompt-tuning** for CLUE classification task。 `<br>` **Path**: applications/benchmark/clue                                                                                                          | BERT, RoBERTa, DeBERTa                  |                                                                               |
|                                  | run_clue_cmrc.sh              | **Goal**: Standard **Fine-tuning** for CLUE CMRC2018 task. `<br>` **Path**: applications/benchmark/cluemrc                                                                                                                                          | BERT, RoBERTa, DeBERTa                  |                                                                               |
|                                  | run_clue_c3.sh                | **Goal**: Standard **Fine-tuning** for CLUE C3 task. `<br>` **Path**: applications/benchmark/cluemrc                                                                                                                                                | BERT, RoBERTa, DeBERTa                  |                                                                               |
|                                  | run_clue_chid.sh              | **Goal**: Standard **Fine-tuning** for CLUE CHID task. `<br>` **Path**: applications/benchmark/cluemrc                                                                                                                                              | BERT, RoBERTa, DeBERTa                  |                                                                               |
| **Instruction-Prompting**  | run_causal_instruction.sh     | **Goal**: **Cross-task training** via generative Instruction-tuning based on causal PLM. `<font color='red'>`**You can use it to train a small ChatGPT**`</font>`. `<br>` **Path**: applications/instruction_prompting/instruction_tuning | GPT2                                    | [click](./documents/instruction_prompting/generative_instruction_tuning.md)      |
|                                  | run_zh_extract_instruction.sh | **Goal**: **Cross-task training** via extractive Instruction-tuning based on Global Pointer model. `<br>` **Path**: applications/instruction_prompting/chinese_instruction                                                                          | BERT, RoBERTa, DeBERTa                  | [click](./documents/instruction_prompting/instruction_tuning.md)                 |
|                                  | run_causal_incontext_cls.sh   | **Goal**: **In-context learning** for user-defined classification tasks. `<br>` **Path**: applications/instruction_prompting/incontext_learning                                                                                                     | GPT-2                                   | [click](./documents/instruction_prompting/incontext_learning_for_cls.md)         |
| **Information Extraction** | run_extractive_unified_ie.sh  | **Goal**: **HugIE**: training a unified chinese information extraction via extractive instruction-tuning. `<br>` **Path**: applications/information_extraction/HugIE                                                                                | BERT, RoBERTa, DeBERTa                  | [click](./documents/information_extraction/HugIE.md)                             |
|                                  | api_test.py                   | **Goal**: HugIE: API test. `<br>` **Path**: applications/information_extraction/HugIE                                                                                                                                                                     | -                                       | [click](./documents/information_extraction/HugIE.md)                             |
|                                  | run_fewnerd.sh                | **Goal**: **Prototypical learning** for named entity recognition, including SpanProto, TokenProto `<br>` **Path**: applications/information_extraction/fewshot_ner                                                                                  | BERT                                    |                                                                               |
| **Code NLU**               | run_clone_cls.sh              | **Goal**: Standard **Fine-tuning** for code clone classification task. `<br>` **Path**: applications/code/code_clone                                                                                                                                | CodeBERT, CodeT5, GraphCodeBERT, PLBART |                                                                               |
|                                  | run_defect_cls.sh             | **Goal**: Standard **Fine-tuning** for code defect classification task. `<br>` **Path**: applications/code/code_defect                                                                                                                              | CodeBERT, CodeT5, GraphCodeBERT, PLBART |                                                                               |

More details of the pre-built applications and settings with the designed models and processors can be found in [HugNLP Documents](./documents/README.md).

# Quick Use

Here we provide an example to show you to quick use HugNLP.
If you want to perform a classification task on user-defined dataset, you can prepare three json files (``train.json``, ``dev.json``, ``test.json``) on a directory. And you can run the script file

> bash ./application/default_applications/run_seq_cls.sh

Before the experiment, you must define the following parameters in the script file ``run_seq_cls.sh``.

- --model_name_or_path: the pre-trained model name or path. e.g. bert-base-uncased
- --data_path: the path of the dataset (including ``train.json``, ``dev.json`` and ``test.json``), e.g. ``./datasets/data_example/cls/``.
- --user_defined: you must define label_names if there is not exist a ``label_names.txt``.

If you want to use prompt-based fine-tuning, you can add the following parameters:

- --use_prompt_for_cls
- ---task_type: one of ``masked_prompt_cls``, ``masked_prompt_prefix_cls``,``masked_prompt_ptuning_cls``, ``masked_prompt_adapter_cls``.

You also should add ``template.json`` and ``label_words_mapping.json``.

If you wang to use parameter-efficient learning, you can add the following parameter:

- --use_freezing

The example of ``run_seq_cls.sh`` is:

```bash
path=chinese-macbert-base
MODEL_TYPE=bert
data_path=/wjn/frameworks/HugNLP/datasets/data_example/cls
TASK_TYPE=head_cls
len=196
bz=4
epoch=10
eval_step=50
wr_step=10
lr=1e-05

export CUDA_VISIBLE_DEVICES=0,1
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=6014 hugnlp_runner.py \
--model_name_or_path=$path \
--data_dir=$data_path \
--output_dir=./outputs/default/sequence_classification\
--seed=42 \
--exp_name=default-cls \
--max_seq_length=$len \
--max_eval_seq_length=$len \
--do_train \
--do_eval \
--do_predict \
--per_device_train_batch_size=$bz \
--per_device_eval_batch_size=4 \
--gradient_accumulation_steps=1 \
--evaluation_strategy=steps \
--learning_rate=$lr \
--num_train_epochs=$epoch \
--logging_steps=100000000 \
--eval_steps=$eval_step \
--save_steps=$eval_step \
--save_total_limit=1 \
--warmup_steps=$wr_step \
--load_best_model_at_end \
--report_to=none \
--task_name=default_cls \
--task_type=$TASK_TYPE \
--model_type=$MODEL_TYPE \
--metric_for_best_model=acc \
--pad_to_max_length=True \
--remove_unused_columns=False \
--overwrite_output_dir \
--fp16 \
--label_names=labels \
--keep_predict_labels \
--user_defined="label_names=entailment,neutral,contradiction"
```

# Quick Develop

This section is for developer.
HugNLP is easy to use and develop. We draw a workflow in the following figure to show how to develop a new running task.

<p align="center">
    <br>
    <img src="images/workflow.png" width="90%"/>
    <br>
</p>
It consists of five main steps, including library installation, data preparation, processor selection or design, model selection or design, and application design.
This illustrates that HugNLP can simplify the implementation of complex NLP models and tasks.

# Demo API Example

Here, we show an example of the pre-built API application: **HugIE: Towards Chinese Unified Information Extraction via Extractive MRC and Instruction-tuning**.

### Introduction:

Information Extraction (IE) aims to extract structure knowledge from un-structure text. The structure knowledge is formed as a triple ""(head_entity, relation, tail_entity)"". IE consists of two main tasks:

- Named Entity Recognition (NER) aims to extract all entity mentions of one type.
- Relation Extraction (RE). It has two kinds of goal, the first aims to classify the relation between two entities, and the second aims to predict the tail entity when given one head entity and the corresponding relation.

### Solutions:

- We unify the tasks of NER and RE into the paradigm of extractive question answering (i.e., machine reading comprehension).
- We design task-specific instruction and language prompts for NER and RE.

> For the NER task:
>
> - instruction: "找到文章中所有【{entity_type}】类型的实体？文章：【{passage_text}】"
>
> For the RE task:
>
> - instruction: "找到文章中【{head_entity}】的【{relation}】？文章：【{passage_text}】"

- During the training, we utilize Global Pointer with Chinese-Macbert as the basic model.；

### Usage:

Our model is saved in Hugging Face: [https://huggingface.co/wjn1996/wjn1996-hugnlp-hugie-large-zh](https://huggingface.co/wjn1996/wjn1996-hugnlp-hugie-large-zh).

Quick use HugIE for Chinese information extraction：

```python
from applications.information_extraction.HugIE.api_test import HugIEAPI
model_type = "bert"
hugie_model_name_or_path = "wjn1996/wjn1996-hugnlp-hugie-large-zh"
hugie = HugIEAPI("bert", hugie_model_name_or_path)
text = "央广网北京2月23日消息 据中国地震台网正式测定，2月23日8时37分在塔吉克斯坦发生7.2级地震，震源深度10公里，震中位于北纬37.98度，东经73.29度，距我国边境线最近约82公里，地震造成新疆喀什等地震感强烈。"

entity = "塔吉克斯坦地震"
relation = "震源位置"
predictions, topk_predictions = hugie.request(text, entity, relation=relation)
print("entity:{}, relation:{}".format(entity, relation))
print("predictions:\n{}".format(predictions))
print("topk_predictions:\n{}".format(predictions))
print("\n\n")

"""
# 事件信息输出结果：
entity:塔吉克斯坦地震, relation:震源位置
predictions:
{0: ["10公里", "距我国边境线最近约82公里", "北纬37.98度，东经73.29度", "北纬37.98度，东经73.29度，距我国边境线最近约82公里"]}
topk_predictions:
{0: [{"answer": "10公里", "prob": 0.9895901083946228, "pos": [(80, 84)]}, {"answer": "距我国边境线最近约82公里", "prob": 0.8584909439086914, "pos": [(107, 120)]}, {"answer": "北纬37.98度，东经73.29度", "prob": 0.7202121615409851, "pos": [(89, 106)]}, {"answer": "北纬37.98度，东经73.29度，距我国边境线最近约82公里", "prob": 0.11628123372793198, "pos": [(89, 120)]}]}
"""

entity = "塔吉克斯坦地震"
relation = "时间"
predictions, topk_predictions = hugie.request(text, entity, relation=relation)
print("entity:{}, relation:{}".format(entity, relation))
print("predictions:\n{}".format(predictions))
print("topk_predictions:\n{}".format(predictions))
print("\n\n")

"""
# 事件信息输出结果：
entity:塔吉克斯坦地震, relation:时间
predictions:
{0: ["2月23日8时37分"]}
topk_predictions:
{0: [{"answer": "2月23日8时37分", "prob": 0.9999995231628418, "pos": [(49, 59)]}]}
"""
```

# Contact

You can contact the author `Jianing Wang` from github.
The interaction group in QQ or dingding will come soon.

# Cite Me

If you find this repository helpful, feel free to cite our paper:

```latex
@misc{wang2023hugnlp,
  doi       = {10.48550/ARXIV.2302.14286},
  url       = {https://arxiv.org/abs/2302.14286},
  author    = {Jianing Wang, Nuo Chen, Qiushi Sun, Wenkang Huang, Chengyu Wang, Ming Gao},
  title     = {HugNLP: A Unified and Comprehensive Library for Natural Language Processing},
  year      = {2023}
}
```

# References

1. Jianing Wang, Wenkang Huang, Minghui Qiu, Qiuhui Shi, Hongbin Wang, Xiang Li, Ming Gao:
   Knowledge Prompting in Pre-trained Language Model for Natural Language Understanding. EMNLP 2022: 3164-3177
2. Chengyu Wang, Jianing Wang, Minghui Qiu, Jun Huang, Ming Gao: TransPrompt: Towards an Automatic Transferable Prompting Framework for Few-shot Text Classification. EMNLP 2021: 2792-2802
3. Jianing Wang, Chengyu Wang, Jun Huang, Ming Gao, Aoying Zhou: Uncertainty-aware Self-training for Low-resource Neural Sequence Labeling. AAAI 2023.

# Acknowledgement

We thank to the Platform of AI (PAI) in Alibaba Group to support our work. The friend framework is [EasyNLP](https://github.com/alibaba/EasyNLP). We also thank all the developers that contribute to our work!
