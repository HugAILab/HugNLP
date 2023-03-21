# Causal LM for Continual Pre-training
## 一、英文Wikipedia语料预训练
### 1.1 语料获取与处理
wikipedia下载语料：[https://dumps.wikimedia.org/enwiki/](https://dumps.wikimedia.org/enwiki/)
使用wikiprocessor工具进行预处理，得到处理后的文本。预训练一般不需要验证集和测试集，但在实际使用中可以自行选择验证和测试。
> 我们提供了几个预训练数据：
> - total_pretrain_data.txt：2600万语料
> - total_pretrain_data_10m.txt：1000万语料
> - total_pretrain_data_10000.txt：1万语料


### 1.2 定义处理器（Processors）
#### CausalLMTextLineProcessor
**位置**：HugNLP/processors/pretraining/causal_lm/data_processor.py
**功能**：读入数据，分词、生成features
**读入数据文件（可自定义名称）**：train.txt、dev.txt、test.txt
**数据格式**：txt文件，其中每行表示一个句子
**额外传入参数**：
无
**DataCollator配置**：

- DataCollatorForCausalLM：生成模型输入Feature张量，

**注意事项**：

- 输入一个文本，自回归生成训练时，默认训练整个序列，且每个token的label为下一个token。输入时只需要直接将input_ids复制一份为label即可。对于padding部分，label需要设置为-100，表示不计算对应的loss。

**源代码**：
```python
"""
Processing data for Causal LM
The pre-training corpus is saved in 'txt' file. Each line is a sentence.
"""
class CausalLMITextLineProcessor(DataProcessor):
    def __init__(self, data_args, training_args, model_args):
        super().__init__(data_args, training_args, model_args)


    def get_data_collator(self):
        return DataCollatorForCausalLM(
            tokenizer=self.tokenizer,
            pad_to_max_length=self.data_args.pad_to_max_length,
        )


    def get_examples(self, set_type=None):
        data_files = {}
        if self.data_args.train_file is not None:
            data_files["train"] = self.data_args.train_file
            extension = self.data_args.train_file.split(".")[-1]
        if self.data_args.validation_file is not None:
            data_files["validation"] = self.data_args.validation_file
            extension = self.data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=self.model_args.cache_dir)
        # raw_datasets['train'] = raw_datasets['train'].shuffle()
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{self.data_args.validation_split_percentage}%]",
                cache_dir=self.model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{self.data_args.validation_split_percentage}%:]",
                cache_dir=self.model_args.cache_dir,
            )
        return raw_datasets


    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[p.label_ids != -100]
        labels = p.label_ids[p.label_ids != -100]
        acc = (preds == labels).mean()
        return {
            'eval_acc': round(acc, 4)
        }


    def get_tokenized_datasets(self):


        data_files = {}
        if self.data_args.train_file is not None:
            data_files["train"] = self.data_args.train_file
            extension = self.data_args.train_file.split(".")[-1]
        if self.data_args.validation_file is not None:
            data_files["validation"] = self.data_args.validation_file
            extension = self.data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=self.model_args.cache_dir)
        # raw_datasets['train'] = raw_datasets['train'].shuffle()
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{self.data_args.validation_split_percentage}%]",
                cache_dir=self.model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{self.data_args.validation_split_percentage}%:]",
                cache_dir=self.model_args.cache_dir,
            )
        logger.info(f'validation fingerprint {raw_datasets}')
        if self.training_args.do_train:
            column_names = raw_datasets["train"].column_names
        else:
            column_names = raw_datasets["validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        max_seq_length = self.tokenizer.model_max_length if self.data_args.max_seq_length is None else self.data_args.max_seq_length
        # When using line_by_line, we just tokenize each nonempty line.
        # padding = "max_length" if self.data_args.pad_to_max_length else False
        padding = False


        tokenizer = self.tokenizer


        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            # examples['length'] = [len(line) for line in examples[text_column_name]]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )


        with self.training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
        return tokenized_datasets
```

### 1.3 定义模型（Models）
定义Causal LM模型，默认选择GPT2进行训练，模型名称为：

- GPT2ForCausalLM：地址：HugNLP/models/language_modeling/causal_lm.py

详细介绍参考：[Causal LM预训练](https://www.yuque.com/wangjianing-jrsey/ktxelv/lfzdpikq7thiwhx4?view=doc_embed)

**超参数设置**：
机器配置：4卡V100（32G）

| 语料数量 | 10k（1万） | 100k（10万） | 1M（100万） | 10M（1000万） |
| --- | --- | --- | --- | --- |
| batch_size |  |  |  |  |
| gradient_accumulation_steps |  |  |  |  |
| learning_rate |  |  |  |  |
|  |  |  |  |  |

### 1.4 应用任务定义（Applications）
**位置**：HugNLP/applications/pretraining/run_pretrain_causal_lm.sh
**应用执行脚本**：
```
path=/wjn/pre-trained-lm/gpt2
# path=/wjn/pre-trained-lm/gpt2-medium
# path=/wjn/pre-trained-lm/gpt2-large
# path=/wjn/pre-trained-lm/gpt2-xl

model_name=gpt2

data_path=/wjn/nlp_task_datasets/wikipedia_corpus

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 6013 nlp_runner.py \
--model_name_or_path=$path \
--data_dir=$data_path \
--train_file=$data_path/total_pretrain_data_10000.txt \
--max_seq_length=512 \
--output_dir=/wjn/frameworks/HugNLP/output/pretrain/casual_lm_$model_name/ \
--do_train \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no \
--save_strategy=steps \
--gradient_accumulation_steps=2 \
--learning_rate=1e-05 \
--logging_steps=10000000 \
--save_steps=1000 \
--save_total_limit=5 \
--num_train_epochs=5 \
--report_to=none \
--task_name=causal_lm_text_line \
--task_type=causal_lm \
--model_type=gpt2 \
--exp_name=causal_lm \
--tracking_uri=runs/ \
--warmup_steps=100 \
--ignore_data_skip \
--remove_unused_columns=False \
--fp16 \
--max_eval_samples=30000 \
--cache_dir=/wjn/.cache \
--overwrite_output_dir
```

**运行效果**：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/12897066/1676619774385-ba374ebe-430c-4f25-930d-358e4d32c3ee.png#averageHue=%230c0c0c&clientId=u9ae83d77-657b-4&from=paste&height=378&id=ue8d12655&name=image.png&originHeight=756&originWidth=1190&originalType=binary&ratio=2&rotation=0&showTitle=false&size=143495&status=done&style=none&taskId=u88be8b2a-e7be-495c-8869-d37813b358c&title=&width=595)
**实验结果**：
