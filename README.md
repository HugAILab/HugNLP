<p align="center">
    <br>
    <img src="images/logo.png" width="360"/>
    <br>
</p>

<p align="center" style="font-size:22px;"> <b> Welcome to use HugNLP. 🤗 Hugging for NLP! </b>
<p>

[![Build Status](https://app.travis-ci.com/nchen909/HugNLP.svg?branch=main)](https://app.travis-ci.com/github/wjn1996/HugNLP)[![GitHub pull-requests](https://img.shields.io/github/issues-pr/wjn1996/HugNLP.svg)](https://github.com/wjn1996/HugNLP/pull/)[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

# About HugNLP

HugNLP is a novel development and application library based on [Hugging Face](https://huggingface.co/) for improving the convenience and effectiveness of NLP researchers. The founder and main developer is [Jianing Wang](https://wjn1996.github.io/). The collaborators (programmers) are [Nuo Chen](https://github.com/nchen909) and [Qiushi Sun](https://github.com/QiushiSun).

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
1. Uncertainty Estimation aims to calculate the model certainty when in semi-supervised learning.
1. We also design Prediction Calibration, which can be used to further improve the accuracy by calibrating the distribution and alleviating the semantics bias problem.

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
1. inference-style paradigm: all the tasks can be viewed as natural language inference to match the relations between inputs and outputs.

### Self-training with Uncertainty Estimation

Self-training can address the labeled data scarcity issue by leveraging the large-scale unlabeled data in addition to labeled data, which is one of the mature paradigms in semi-supervised learning.
However, the standard self-training may generate too much noise, inevitably degrading the model performance due to confirmation bias.
Thus, we present uncertainty-aware self-training. Specifically, we train a teacher model on few-shot labeled data, and then use Monte Carlo (MC) dropout technique in Bayesian neural network (BNN) to approximate the model certainty, and judiciously select the examples that have a higher model certainty of the teacher.

### Parameter-Efficient Learning

To improve the training efficiency of HugNLP, we also implement parameter-efficient learning, which aims to freeze some parameters in the backbone so that we only tune a few parameters during model training.
We develop some novel parameter-efficient learning approaches, such as Prefix-tuning, Adapter-tuning, BitFit and LoRA, etc.

# Quick Use

> git clone https://github.com/wjn1996/HugNLP.git
>
> cd HugNLP
>
> python3 setup.py install

At present, the project is still being developed and improved, and there may be some `bugs` in use, please understand. We also look forward to your being able to ask issues or committing some valuable pull requests.

# Quick Develop

HugNLP is easy to use and develop. We draw a workflow in the following figure to show how to develop a new running task.

<p align="center">
    <br>
    <img src="images/workflow.png" width="90%"/>
    <br>
</p>
It consists of five main steps, including library installation, data preparation, processor selection or design, model selection or design, and application design.
This illustrates that HugNLP can simplify the implementation of complex NLP models and tasks.

# Demo Example

## HugIE: Towards Chinese Unified Information Extraction via Extractive MRC and Instruction-tuning

### Introduction

Information Extraction (IE) aims to extract structure knowledge from un-structure text. The structure knowledge is formed as a triple ""(head_entity, relation, tail_entity)"". IE consists of two main tasks:

- Named Entity Recognition (NER) aims to extract all entity mentions of one type.
- Relation Extraction (RE). It has two kinds of goal, the first aims to classify the relation between two entities, and the second aims to predict the tail entity when given one head entity and the corresponding relation.

### Solutions

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

### Usage

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

# References

```latex
@misc{wang2023hugnlp,
  doi       = {10.48550/ARXIV.2302.14286},
  url       = {https://arxiv.org/abs/2302.14286},
  author    = {Wang, Jianing and Chen, Nuo and Sun, Qiushi and Huang, Wenkang and Wang, Chengyu and Gao, Ming},
  title     = {HugNLP: A Unified and Comprehensive Library for Natural Language Processing},
  publisher = {arXiv},
  year      = {2023}
}
```

# Acknowledgement

We thank to the Platform of AI (PAI) in Alibaba Group to support our work. The friend framework is [EasyNLP](https://github.com/alibaba/EasyNLP). We also thank all the developers that contribute to our work!
