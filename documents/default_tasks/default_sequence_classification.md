# Default Sequence Classification
### 一、数据准备
**训练数据文件**：train.json、dev.json、test.json
**数据格式**：
```cpp
{
    "sentence1": "xx",
    "sentence2": "xx",
    "label": "xx",
}
```
**标签集合**：
有两种传入标签集合的方式：

- 通过自定义参数传入的形式（优先级最高）：定义 user_defined="label_names=xx,xx"。
- 创建一个文件名label_names.txt，文件的每一行存放label名称；

### 二、Models（模型）
可选择任意一个分类模型，包括：
```python
"auto_cls": AutoModelForSequenceClassification, # huggingface cls
    "classification": AutoModelForSequenceClassification, # huggingface cls
    "head_cls": {
        "bert": BertForSequenceClassification,
        "roberta": RobertaForSequenceClassification,
        "bart": BartForSequenceClassification,
    }, # use standard fine-tuning head for cls, e.g., bert+mlp
    "head_prefix_cls": {
        "bert": BertPrefixForSequenceClassification,
        "roberta": RobertaPrefixForSequenceClassification,
    }, # use standard fine-tuning head with prefix-tuning technique for cls, e.g., bert+mlp
    "head_ptuning_cls": {
        "bert": BertPtuningForSequenceClassification,
        "roberta": RobertaPtuningForSequenceClassification,
    }, # use standard fine-tuning head with p-tuning technique for cls, e.g., bert+mlp
    "head_adapter_cls": {
        "bert": BertAdapterForSequenceClassification,
        "roberta": RobertaAdapterForSequenceClassification,
    }, # use standard fine-tuning head with adapter-tuning technique for cls, e.g., bert+mlp
```
### 三、Processors
位置：HugNLP/processors/default_task_processors/data_processor.py

### 四、Application
定义脚本。

位置：HugNLP/applications/default_applications/run_seq_cls.sh
需要自定义的参数（非调参部分）：

- --path：模型的存储位置，或Hugging Face模型路径
- --data_path：用户自定义的数据集存放位置
- --user_defined：可选，当data_path下没有label_names.txt文件时，需要人工定义一下标签集合。例如三分类任务的格式为--user_defined="label_names=entailment,neutral,contradiction"
> 注意：user_defined中，多个参数以空格相隔，每个参数的取值以“=”作为连接，label name的读取则以“,”相隔。


可通过改变传入参数的值来选择相应的训练方法。
### 4.1 使用标准Fine-tuning
普通训练模式

- --task_type可选“auto_cls”、“classification”、“head_cls”；

使用参数有效性训练：

- --task_type可选“head_prefix_cls”、“head_ptuning_cls”、“head_adapter_cls”
- --use_freezing

样例：
```bash
#### pre-trained lm path
path=/wjn/pre-trained-lm/chinese-macbert-base/
MODEL_TYPE=bert

#### task data path (use should change this path)
data_path=/wjn/frameworks/HugNLP/datasets/data_example/cls

len=196
bz=4 # 8
epoch=10
eval_step=50
wr_step=50
lr=3e-05

export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 --master_port=6014 hugnlp_runner.py \
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
--task_type=head_cls \
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

### 4.1 使用Prompt-Tuning
在数据准备阶段，在数据集目录下存放如下两个文件：
```json
[{"prefix_template": "", "suffix_template": ""}, {"prefix_template": "<mask> <mask>", "suffix_template": ""}]
```
template.json中是一个数组，数组中的第一个元素表示对第一个句子的开头和结尾的模板，同理，数组中的第二个元素表示对第二个句子的开头和结尾的模板。
```json
{"entailment": ["蕴含"], "neutral": ["中立"], "contradiction": ["矛盾"]}
```
label_words_mapping.json中保存每一个类别名称对应的一个label word。
> 注意：
> - template中的masked token必须使用“<mask>”；
> - label名称需要与label_name.txt或传入的参数保持一致，label word的词个数需要与<mask>的数量保持一致。

需要显式添加下列参数：

- --use_prompt_for_cls
- --task_type可选“masked_prompt_cls”、“masked_prompt_prefix_cls”、“masked_prompt_ptuning_cls”、“masked_prompt_adapter_cls”；

使用参数有效性训练：

- --task_type可选“head_prefix_cls”、“head_ptuning_cls”、“head_adapter_cls”
- --use_freezing
