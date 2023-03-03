# Instruction-tuning 技术文档


---

### 基于MRC的中文instruction-tuning
（1）方法介绍
xxx

（2）数据准备

数据保存为json格式。每一行为一个json字符串，对应一个样本example，格式如下所示（必须包含如下所列出的所有key）：
```json
{
    "ID": "xxx",
    "instruction": "xxxx",
    "target": "xxx",
    "data_type": "xx",
    "start": 0,
}
```
- "ID"：表示对应一条样本的编号；
- "instruction"：表示根据设计的模板、原始文本生成的instruction。如果是分类任务，则还有相应的候选label；
- "target"：表示该样本的标签。如果是分类任务，则为分类标签名，如果是抽取式任务，则为抽取的结果，如果是生成任务，则为生成的结果。特别地，如果是多类分类、实体识别、多区间MRC任务，target应为列表；
- "data_type"：样本原始对应的任务类型，例如classification、ner、mrc等；
- "start"：表示正确答案在instruction中的字符索引。特别地，如果是多类分类、实体识别、多区间MRC任务，start应为列表。
> 我们希望使用MRC的方式实现多任务统一训练，因此，需要将各类任务（例如分类、NER、QA）等转换为抽取式MRC的形式。

其他可选key，下列key只用于描述当前样本原始的信息，instruciton-tuning并不使用：
- "context"：表示该样本对应的文本，如果是sentence pair任务，则直接拼接；
- "question": 对于QA类型任务，表示问题；
- "label": 对于分类任务，表示标签名；
- "verbalizer": 对于分类任务，所有类别拼接，拼接间隔字符串为“||”；
- "entities": 对于实体识别任务，则列出所有实体；

（3）训练与推理instruction-tuning
