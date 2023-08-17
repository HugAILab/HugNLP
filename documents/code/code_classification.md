# Code Clone&Defect Detection

### 一、数据准备

#### 默认格式

**训练数据文件**：train.json、dev.json、test.json

**Clone Detection数据格式**：

```
{
	"label": "0",
	"func1":"xx",
	"func2":"xx",
	"id": xx
}

```

**Defect Detection数据格式**：

```cpp
{
	"label": "0",
	"func1":"xx",
	"id": xx
}
```

#### 外源数据集

你也可以从[CodeXGLUE](https://github.com/microsoft/CodeXGLUE)等外源数据集中自行导入数据，并将数据按说明转换为默认格式。或者，使用以下代码下载到/data/to/codexglue：

```bash
cd /data/to/codexglue
pip install gdown
gdown https://drive.google.com/uc?export=download&id=1BBeHFlKoyanbxaqFJ6RRWlqpiokhDhY7
unzip data.zip
rm data.zip
```

并在application中指定data_path。

```bash
#### task data path (use should change this path)
data_path=/data/to/codexglue
```

### 二、Models（模型）

支持的代码预训练模型包括：

```python
    "code_cls": {
        "roberta": RobertaForCodeClassification,
        "codebert": CodeBERTForCodeClassification,
        "graphcodebert": GraphCodeBERTForCodeClassification,
        "codet5": CodeT5ForCodeClassification,
        "plbart": PLBARTForCodeClassification,
    },
```

### 三、Processors

位置：HugNLP/processors/code/code_{clone,defect}/data_processor.py

### 四、Application

定义脚本。

位置：HugNLP/applications/code/code_{clone,defect}/run\_{clone,defect}\_cls.sh
需要自定义的参数（非调参部分）：

- --path：模型的存储位置，或Hugging Face模型路径
- --MODEL_TYPE：代码预训练模型，可选范围见Models节
- --data_path：用户自定义的数据集存放位置
- --user_defined：可选，当data_path下没有label_names.txt文件时，需要人工定义一下标签集合，并且在代码分类任务中，建议指定最大序列长度max_target_length。例如克隆检测分类任务的格式为--user_defined="label_names=0,1 max_target_length=512"

> 注意：user_defined中，多个参数以空格相隔，每个参数的取值以“=”作为连接，label name的读取则以“,”相隔。

可通过改变传入参数的值来选择相应的训练方法。

### 4.1 Clone Detection

克隆检测 (Clone Detection) 样例：

```bash
#### pre-trained lm path
path=/code/cn/CodePrompt/data/huggingface_models/codebert-base/
MODEL_TYPE=codebert

#### task data path (use should change this path)
data_path=/code/cn/HugNLP/datasets/data_example/clone/

TASK_TYPE=code_cls
# TASK_TYPE=masked_prompt_prefix_cls

len=512
bz=4 # 8
epoch=10
eval_step=50
wr_step=10
lr=1e-05


export CUDA_VISIBLE_DEVICES=0,1
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=6014 hugnlp_runner.py \
--model_name_or_path=$path \
--data_dir=$data_path \
--output_dir=./outputs/code/clone_classification_codebert\
--seed=1234 \
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
--task_name=code_clone \
--task_type=$TASK_TYPE \
--model_type=$MODEL_TYPE \
--metric_for_best_model=acc \
--pad_to_max_length=True \
--remove_unused_columns=False \
--overwrite_output_dir \
--label_names=labels \
--keep_predict_labels \
--user_defined="label_names=0,1 max_target_length=512" \
```

### 4.1 Defect Detection

缺陷检测 (Defect Detection) 样例：

```bash
#### pre-trained lm path
path=/code/cn/CodePrompt/data/huggingface_models/codebert-base/
MODEL_TYPE=codebert

#### task data path (use should change this path)
data_path=/code/cn/HugNLP/datasets/data_example/defect/

TASK_TYPE=code_cls
# TASK_TYPE=masked_prompt_prefix_cls

len=512
bz=4 # 8
epoch=10
eval_step=50
wr_step=10
lr=1e-05


export CUDA_VISIBLE_DEVICES=0,1
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=6014 hugnlp_runner.py \
--model_name_or_path=$path \
--data_dir=$data_path \
--output_dir=./outputs/code/defect_classification_codebert\
--seed=1234 \
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
--task_name=code_defect \
--task_type=$TASK_TYPE \
--model_type=$MODEL_TYPE \
--metric_for_best_model=acc \
--pad_to_max_length=True \
--remove_unused_columns=False \
--overwrite_output_dir \
--label_names=labels \
--keep_predict_labels \
--user_defined="label_names=0,1 max_target_length=3" \

```
