<a name="UdTtU"></a>
# GPT-style PLMs In-Context Learning for Sequence Classification
OpenAI在2020年发布的GPT-3模型中提出了新的概念叫做**In-Context Learning（ICL）**，其旨在挑选少量的标注样本作为提示（Prompt），使得**无需参数更新**的条件下即可激发大语言模型生成所要的结果。总的来说，ICL具备如下性质：

- 只需要少量标注样本作为提示；
- 无需训练模型，直接通过模型生成获得结果；

In-Context Learning可以完成分类和生成两种任务。HugNLP为此实现基于GPT-family模型的In-Context Learning的Application并分别用于分类和生成任务上。
<a name="m1JWE"></a>
### 一、基于In-Context Learning的文本分类
基于ICL的分类样例如下图所示：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/12897066/1679374196765-09b03064-b86f-4fae-b6d2-b32bdca040c6.png#averageHue=%23e9e9dd&clientId=ud0fe072c-aa38-4&from=paste&height=242&id=u91f78124&originHeight=484&originWidth=830&originalType=binary&ratio=2&rotation=0&showTitle=false&size=76646&status=done&style=none&taskId=u25036381-f558-4c0b-91cd-c8107108353&title=&width=415)
给定 $K$标注样本 $D=\{(x_i, y_i)\}_{i=1}^{K}$以及一个测试样本 $x_{test}$，每个标注样本包括输入句子 $x_i$和对应的标签 $y_i$。通过模板 $\mathcal{P}$将这些样本拼接成为一个Prompt，记作 $P=\mathcal{P}(D, x_{test})$。例如上图的例子，$K=3$，并在每个输入句子和标签之间插入换行符“\n”。最后喂入GPT系列模型中，生成出结果。

由于是分类任务，我们需要获得每个类别标签对应的概率。因此我们采用Prompt-tuning中的**Verbalizer**实现。Verbalizer可以简单描述为标签词对类别的映射关系。例如在情感分析中，“great”可以映射为“positive”类别，而“bad”可以映射为“negative”类别。当GPT模型生成出一些结果时，我们可以获得标签词对应的概率来代表对应类别的概率。

下面介绍使用HugNLP开发基于In-Context Learning的分类应用，并介绍如何使用。

<a name="MVwnT"></a>
#### 1.1 数据与格式
指定数据目录，该目录需要存在如下文件，如图所示：


<p align="center">
    <img src="https://cdn.nlark.com/yuque/0/2023/png/12897066/1679375443888-4da35763-517d-4315-9df9-2709f310a9ab.png#averageHue=%23262627&clientId=u58c3411b-1d62-4&from=paste&height=155&id=u11214516&originHeight=310&originWidth=374&originalType=binary&ratio=2&rotation=0&showTitle=false&size=32233&status=done&style=none&taskId=udcb08971-8a45-4288-bb95-7e54757e93e&title=&width=187" width="150"/>
<p>

**（1）train.json、dev.json和test.json为**数据集文件，每一行为一条数据，需要包含“sentence1”和“label”两个键，（如果是匹配任务，需要有“sentence2”）。数据格式样例如下所示：
> **Single-sentence任务**：

```json
{"sentence1": "a joyous occasion", "label": "1"}
```
> **Sentence-pair任务**：

```json
{"sentence1": "a joyous occasion", "sentence2": "a great occasion", "label": "1"}
```
**（2）label_names.json文件**：保存当前任务数据集的所有类别及其描述。格式样例如下：
```json
{
    "0": "Negative",
    "1": "Positvie"
}
```

- 键：表示数据集给定的类别
- 值：表示当前类别对应的解释描述。

（**3）label_words_mapping.json文件**：保存每个类别对应的标签词，样例如下：
```json
{"0": ["bad"], "1": ["grate"]}
```

- 键：数据集给定的类别，需要与label_names.json中的键保持一致
- 值：标签词数组，保存对应类别的标签词

**（4）instruction.json文件：**保存该任务的指令，样例如下：
```json
{"instruction": "Classify the sentiment text.", "input_prompt": "Review: ", "output_prompt": "Sentiment: "}
```

- instruction：任务指令，用来描述当前任务要做什么事情，以及一些信息。；
- input_prompt：每个样本的输入句子前的提示；
- output_prompt：每个样本的输出前的提示；

**（5）template.json为模板文件**：保存当前任务数据集的模板，格式如下：
```json
[{"prefix_template": "", "suffix_template": ""}, {"prefix_template": "<mask> <mask>", "suffix_template": ""}]
```
该文件只有一行列表，列表中有两个字典，分别表示第一个句子和第二个句子的模板（不论是single-sentence任务还是sentence-pair任务，都需要包含两个“{"prefix_template": "", "suffix_template": ""}”字典）。对于每个字典，其参数意义如下：

- prefix_template：句子前缀模板；
- suffix_template：句子后缀模板；

在In-Context Learning场景下，这两个参数有时候与instruction.json中的input_prompt和output_prompt一样。
例如如果输入的样本为：
> {"sentence1": "a joyous occasion", "label": "1"}

例如如果文件定义为如下所示：
> [{"prefix_template": "Sentiment: ", "suffix_template": "Label: "}, {"prefix_template": "", "suffix_template": ""}]

那么通过模板得到的样本变为：
> Sentiment: a joyous occasion. Label: Positive.

<a name="HYwLu"></a>
#### 1.2 Processor定义
位置：HugNLP/processors/instruction_prompting/incontext_learning/data_processor.py
指定超参数：user_defined参数，需包含如下两个参数：

- data_name（可选）：当前数据集的名称；
- num_incontext_example（必选）：In-Context Example的数量，即 $K$大小；
- l（“L”的小写，必选）：表示希望GPT模型生成的token数量。分类任务中默认为1。
- use_calibrate（可选）：是否采用calibrate对预测的结果进行校准（参考论文Calibrate Before Use）

主要流程：

- 读取训练集、验证集和测试集数据；
- 从训练数据中进行采样 $K$个样本作为in-context example；
- 读取每个验证集或测试集样本，将其与采样的 $K$个标注样本喂入到**InstructionPromptProcessor**（位置：HugNLP/processors/basic_processors/prompt_processor.py**）**中构建In-Context Learning模板。
- 喂入GPT（例如GPT-2）模型中；
- 通过CausalSequenceClassificationEvaluator（位置：HugNLP/evaluators/sequence_classification_evaluator.py）生成结果，并根据label_word_mapping获得每个类标签的概率；
- 最后完成测试评估。

<a name="RLyo6"></a>
#### 1.3 Model
默认情况下模型采用GPT2模型，位置：HugNLP/models/sequence_classification/causal_prompt_cls.py

<a name="GcENL"></a>
#### 1.4 Application
定义Application脚本，位置：HugNLP/applications/instruction/incontext_learning/run_causal_incontext_cls.sh

```json
#### pre-trained lm path
path=/wjn/pre-trained-lm/gpt2-xl
MODEL_TYPE=gpt2

#### task data path (user should change this path)
data_path=./datasets/data_example/incontext_cls

export CUDA_VISIBLE_DEVICES=4
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=6020 hugnlp_runner.py \
  --model_name_or_path=$path \
  --data_dir=$data_path\
  --output_dir=./outputs/instruction/incontext_learning \
  --seed=42 \
  --exp_name=gpt2-incontext-cls \
  --max_seq_length=512 \
  --max_eval_seq_length=512 \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size=1 \
  --per_device_eval_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --evaluation_strategy=steps \
  --logging_steps=100000000 \
  --eval_steps=1 \
  --save_steps=1 \
  --save_total_limit=1 \
  --load_best_model_at_end \
  --report_to=none \
  --task_name=causal_incontext_cls \
  --task_type=causal_prompt_cls \
  --model_type=$MODEL_TYPE \
  --metric_for_best_model=macro_f1 \
  --pad_to_max_length=True \
  --remove_unused_columns=False \
  --overwrite_output_dir \
  --label_names=short_labels \
  --keep_predict_labels \
  --cache_dir=/wjn/.cache \
  --user_defined="num_incontext_example=4 l=1 use_calibrate=True" \
  --use_prompt_for_cls
```
评测结果样例：
Calibrate校准前：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/12897066/1680098810923-11ae52eb-177b-4027-80b8-1aab195df22b.png#averageHue=%23202020&clientId=uc34f5563-049c-4&from=paste&height=229&id=uff2f0610&originHeight=458&originWidth=1216&originalType=binary&ratio=2&rotation=0&showTitle=false&size=94949&status=done&style=none&taskId=uc43b414b-502a-42a0-a872-ddafe8f7061&title=&width=608)
Calibrate校准后：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/12897066/1680098496003-5166f392-f1d4-4a7e-b758-e73fd7cd210e.png#averageHue=%231d1d1d&clientId=uc34f5563-049c-4&from=paste&height=290&id=ua390d931&originHeight=580&originWidth=1214&originalType=binary&ratio=2&rotation=0&showTitle=false&size=123915&status=done&style=none&taskId=u314848c6-68b6-40d0-b8d9-6f4433328cc&title=&width=607)

