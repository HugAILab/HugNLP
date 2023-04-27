# HugIE: A Unified Chinese Information Extraction Framework via Extractive Instruction Prompting
## 一、方法介绍
**信息抽取（Information Extraction）** 旨在从非结构化的文本中抽取出结构化信息，是构建知识库的重要步骤之一。信息抽取还常用于事件抽取，捕捉文本中的事件信息。通常这种结构化的信息是以三元组形式存在，其包括头实体、关系和尾实体。例如三元组（中国，首都，北京）可以理解为“中国”与“北京”的关系是“实体”，也可以理解为“中国”的“首都”是“北京”。因此三元组可以很好地描述一个结构化的知识。信息抽取则是从文本中挖掘出这些三元组。

通常信息抽取主要包括两种任务：

- **命名实体识别（Named Entity Recognition）**：识别出文本中指定类型的实体；
- **关系抽取（Relation Extraction）**：给定头尾实体获得其关系，亦或是给定一个头实体和关系，预测对应的尾实体。在事件抽取中，通常定义为后者；

传统的信息抽取方法是将上述两个过程单独完成，即现对文本中识别出所有可能的实体，然后对识别出的实体两两预测关系，从而得出所有三元组，这种基于pipeline的形式往往导致抽取的质量较差，且容易造成误差传播。为了解决这个问题，我们期望提出一种端到端的信息抽取框架，即只需要一个模型即可以同时解决实体识别和关系抽取。

我们研发一款HugIE框架，旨在实现统一信息处理。其主要核心包括如下几个部分：

- 将实体识别和关系抽取，统一为新的范式——基于抽取式阅读理解的方法；
- 定义Instruction Prompt，指导模型生成需要抽取的内容；
- 采用多任务训练的方法训练；

#### 基于抽取式阅读理解的范式统一
传统的实体识别采用序列标注的方法完成，而关系抽取则是文本分类任务，两种类型的任务范式不统一。为了让两种类型的任务实现范式统一，我们采用Instruction-tunin的思想来实现。
> Instruction-tuning是面向GPT-3的生成式模型，只需要通过设计任务的指令模板，即可以诱导模型生成出与任务相关的输出结果。因此所有任务建模为文本生成，不同的任务维护不同的模板即可。

我们将实体识别和关系抽取统一建模为**区间抽取（Span Extraction）**，类似抽取式问答的方法。对于实体识别，我们需要定义Question和Passage：

- Question：提出问题，例如“找到所有人名类型的实体？”；
- Passage：输入的文本，例如“当主持人吴宗宪看到这首歌曲的曲谱后，就邀请周杰伦到阿尔发音乐公司担任音乐助理”。

因此，将Question和Passage喂入抽取式模型中，可以得到预测的区间，例如[3,5]对应“吴宗宪”，[21, 23]对应“周杰伦”。
> 指令：找到文章中所有【{}】类型的实体？文章：【{}】


对于关系抽取，我们将其定义为给定头实体和指定的关系来预测尾实体，因此依然可以转换为抽取式问答的形式。例如头实体是“阿尔发音乐公司”，关系是“音乐助理”：

- Question：“阿尔发音乐公司的音乐助理是？”
- Passage：“当主持人吴宗宪看到这首歌曲的曲谱后，就邀请周杰伦到阿尔发音乐公司担任音乐助理”。

可以得到预测的区间为[21, 23]对应“周杰伦”。
> 指令：找到文章中【{}】的【{}】？文章：【{}】


## 二、数据获取
我们搜集了大量的中文信息抽取语料，包括实体识别、关系抽取、联合实体关系抽取等。可参考的语料如下所示：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/12897066/1677139687397-b774fa3e-3e34-4cc1-9c94-03ca1ded2c9e.png#averageHue=%23f2f2f2&clientId=uf7291b7e-0816-4&from=paste&height=549&id=u32b2f5e7&name=image.png&originHeight=1098&originWidth=972&originalType=binary&ratio=2&rotation=0&showTitle=false&size=344610&status=done&style=none&taskId=ue789fbf9-e585-48fa-8740-053a7b9a7d7&title=&width=486)
![image.png](https://cdn.nlark.com/yuque/0/2023/png/12897066/1677139699964-ec2e5fb5-792f-440c-9d29-85a3e57e2692.png#averageHue=%23f7f7f7&clientId=uf7291b7e-0816-4&from=paste&height=482&id=u26f4bd8b&name=image.png&originHeight=964&originWidth=946&originalType=binary&ratio=2&rotation=0&showTitle=false&size=334745&status=done&style=none&taskId=u209f54fe-22f9-42bd-8b9b-138b5a6e432&title=&width=473)
由于采用的是抽取式问答范式，为了提高模型的泛化能力，我们依然可以直接获得抽取式问答本身的语料，例如CMRC2018等。同时也可以将文本分类、文本匹配等其他自然语言理解任务转换为抽取式问答的形态。
![image.png](https://cdn.nlark.com/yuque/0/2023/png/12897066/1677139803947-c178f87e-7235-4681-a4d8-75e22105378f.png#averageHue=%23f6f6f6&clientId=uf7291b7e-0816-4&from=paste&height=103&id=u4b6eac7e&name=image.png&originHeight=206&originWidth=876&originalType=binary&ratio=2&rotation=0&showTitle=false&size=52075&status=done&style=none&taskId=ubcdc13e7-b304-4c9b-8b21-76e64ea89ed&title=&width=438)
![image.png](https://cdn.nlark.com/yuque/0/2023/png/12897066/1677139819843-8bca6166-f6b2-4be9-9089-6a3f8b1b6c21.png#averageHue=%23f5f5f5&clientId=uf7291b7e-0816-4&from=paste&height=642&id=u3b4ea0de&name=image.png&originHeight=1284&originWidth=942&originalType=binary&ratio=2&rotation=0&showTitle=false&size=389061&status=done&style=none&taskId=u3605369c-b390-41d6-9914-82fbd00e5fb&title=&width=471)
![image.png](https://cdn.nlark.com/yuque/0/2023/png/12897066/1677139831408-8d74634d-dc73-4b81-aefb-dd828cf3cc54.png#averageHue=%23ebebeb&clientId=uf7291b7e-0816-4&from=paste&height=257&id=u70c55c63&name=image.png&originHeight=514&originWidth=944&originalType=binary&ratio=2&rotation=0&showTitle=false&size=193978&status=done&style=none&taskId=ud1a038a0-f190-4f52-9079-b345cf3b067&title=&width=472)
一些比赛数据：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/12897066/1677139845158-3bda1a73-d88e-4b29-8f71-19d1e87ef33a.png#averageHue=%23f4f4f4&clientId=uf7291b7e-0816-4&from=paste&height=290&id=ud3a2bdec&name=image.png&originHeight=580&originWidth=926&originalType=binary&ratio=2&rotation=0&showTitle=false&size=136331&status=done&style=none&taskId=ubdea7ff1-3cb5-43d0-b159-c2942b3a66f&title=&width=463)
针对信息抽取任务，我们设计了统一的数据格式，根据这个格式实现范式统一。结构如下：

- ID：数据编号；
- context：原始的文本；
- entity_type：当为实体识别时，表示该样本中期望抽取的实体类型；当为关系抽取时，可忽略；
- relation_type（可为空）：当不为空时，则对应关系抽取；
- instruction：指令模板，对于实体识别，指令默认为“找到文章中【<entity_type>】类型的实体？”；对于关系抽取，指令模板为找到文章中【<entity>】的【<relation>】？
- data_type：表明任务原始的类型，例如“ner”；
- verbalizer：主要用于分类任务，此时不需要赋值；
- entities：当为实体识别时，列举出所有满足entity_type的实体文本；
- start：期望模型输出的结果的区间起始位置；
- target：正确的输出结果

形成train.json、dev.json等数据文件。样例如下所示：
```
[
  {
    "ID": "DuIE2.0-train-27627",
    "context": "《青春烈火》原名雅典娜女神，是由文化中国、强视传媒、博海影视、博纳影业联合出品的一部民国激战年代情感大剧，由叶璇、刘恩佑、莫小棋、巫迪文李蓓蕾领衔主演，著
  名动作导演谭俏执导，故事背景发生在1932年的上海租界，讲述了一位以“雅典娜”为代号的“叛谍狂花”游走于国仇与家恨之间浴血抗战的传奇故事",
    "entity_type": "影视作品",
    "relation_type": "导演",
    "instruction": "找到文章中【雅典娜女神】的【导演】？文章：【《青春烈火》原名雅典娜女神，是由文化中国、强视传媒、博海影视、博纳影业联合出品的一部民国激战年代情感大剧，
  由叶璇、刘恩佑、莫小棋、巫迪文李蓓蕾领衔主演，著名动作导演谭俏执导，故事背景发生在1932年的上海租界，讲述了一位以“雅典娜”为代号的“叛谍狂花”游走于国仇与家恨之间浴血抗战的传
  奇故事】",
    "verbalizer": "",
    "data_type": "ner",
    "entities": [
      "谭俏"
    ],
    "start": [
      [
        104
      ]
    ],
    "target": [
      "谭俏"
    ]
  },
  ...
]
```
所有数据汇总形成train.json和dev.json。
## 二、定义Processor
#### ChineseExtractiveInstructionProcessor
**位置**：HugNLP/processors/instruction_prompting/chinese_extractive_instruction/data_processor.py
**功能**：读入数据，分词、生成features
**读入数据文件（可自定义名称）**：train.json、dev.json、test.json
**数据格式**：json文件
**额外传入参数**：
无
**DataCollator配置**：

- DataCollatorForGlobalPointer：生成模型输入Feature张量，

**注意事项**：

- 需要tokenizer生成offset_mapping，即分词前后每个token的索引需要有一一映射关系。

## 三、定义模型
将信息抽取建模为MRC任务，可以采用多种区间抽取式模型，HugIE主要运用GlobalPointer模型，位置HugIE/models/span_extraction/global_pointer.py：

- RawGlobalPointer
- BertForEffiGlobalPointer
- RobertaForEffiGlobalPointer
- RoformerForEffiGlobalPointer
- MegatronForEffiGlobalPointer

面向中文领域，HugIE选择chinese-mac-bert作为基础的backbone，并在此基础上进行continual pre-training。MRC的解码器选择BertForEffiGlobalPointer模型。

## 四、应用任务定义
定义应用脚本：
```bash
path=/wjn/pre-trained-lm/chinese-macbert-large # 89
#path=/wjn/pre-trained-lm/chinese_pretrain_mrc_macbert_large
#path=/wjn/pre-trained-lm/chinese-pert-large
#path=/wjn/pre-trained-lm/chinese-pert-large-mrc
# path=/wjn/pre-trained-lm/chinese-roberta-wwm-ext-large # 78
#path=/wjn/pre-trained-lm/chinese_pretrain_mrc_roberta_wwm_ext_large
#path=/wjn/pre-trained-lm/chinesebert-large
#path=/wjn/pre-trained-lm/structbert-large-zh
#path=/wjn/pre-trained-lm/Erlangshen-MegatronBert-1.3B

# data_path=/wjn/nlp_task_datasets/zh_instruction
data_path=/wjn/nlp_task_datasets/information_extraction/extractive_unified_ie/

export CUDA_VISIBLE_DEVICES=0,1
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=6019 hugnlp_runner.py \
--model_name_or_path=$path \
--data_dir=$data_path \
--output_dir=./outputs/information_extraction/extractive_unified_ie_test/ \
--seed=42 \
--exp_name=unified-ie-wjn \
--max_seq_length=512 \
--max_eval_seq_length=512 \
--do_train \
--do_eval \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=64 \
--gradient_accumulation_steps=1 \
--evaluation_strategy=steps \
--learning_rate=2e-05 \
--num_train_epochs=3 \
--logging_steps=100000000 \
--eval_steps=500 \
--save_steps=500 \
--save_total_limit=1 \
--warmup_steps=200 \
--load_best_model_at_end \
--report_to=none \
--task_name=zh_mrc_instruction \
--task_type=global_pointer \
--model_type=bert \
--metric_for_best_model=macro_f1 \
--pad_to_max_length=True \
--remove_unused_columns=False \
--overwrite_output_dir \
--fp16 \
--label_names=short_labels \
--keep_predict_labels \
--cache_dir=/wjn/.cache
# --do_adv
```
