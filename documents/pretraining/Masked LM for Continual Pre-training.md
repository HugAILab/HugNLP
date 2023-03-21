# Masked LM for Continual Pre-training
## 一、英文Wikipedia语料预训练
### 1.1 语料获取与处理
wikipedia下载语料：[https://dumps.wikimedia.org/enwiki/](https://dumps.wikimedia.org/enwiki/)
使用wikiprocessor工具进行预处理，得到处理后的文本。预训练一般不需要验证集和测试集，但在实际使用中可以自行选择验证和测试。
> 我们提供了几个预训练数据：
> - total_pretrain_data.txt：2600万语料
> - total_pretrain_data_10m.txt：1000万语料
> - total_pretrain_data_10000.txt：1万语料

### 1.2 定义处理器（Processors）
#### MLMTextLineProcessor
**位置**：HugNLP/processors/pretraining/mlm/data_processor.py
**功能**：读入数据，分词、生成features
**读入数据文件（可自定义名称）**：train.txt、dev.txt、test.txt
**数据格式**：txt文件，其中每行表示一个句子
**额外传入参数**：

- (DataTrainingArguments) mlm_probability：mask token的比例，取值范围为0～1，默认为0.15。

**DataCollator配置**：

- DataCollatorForMaskedLM：生成模型输入Feature张量，
- DataCollatorForMaskedLMWithoutNumber：生成模型输入Feature张量，不考虑文本中数字部分

**注意事项**：

- MLM在预训练时需要构建labels，对于mask位置，label为被mask的token本身，其余的token部分，label设置为-100（cross-entropy对于label为-100时，不计算其loss）

**源代码**：
```python
class MLMTextLineProcessor(DataProcessor):
    def __init__(self, data_args, training_args, model_args):
        super().__init__(data_args, training_args, model_args)

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.data_args.line_by_line and self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForMaskedLMWithoutNumber(
            tokenizer=self.tokenizer,
            mlm_probability=self.data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
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
        padding = "max_length" if self.data_args.pad_to_max_length else False

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
采用Masked Language Modeling进行预训练，可选择几种不同的MLM模型，包括：

- BertForMaskedLM
- RobertaForMaskedLM
- AlbertForMaskedLM
- RoFormerForMaskedLM

可自行设计和搭建MLM的其他模型结构，模型搭建文档详见：[Masked LM预训练](https://www.yuque.com/wangjianing-jrsey/ktxelv/meublgy9pnbfmbbp?view=doc_embed)
### 1.4 应用任务定义（Applications）
**位置**：HugNLP/applications/pretraining/run_pretrain_mlm.sh
**应用执行脚本**：
```
path=/wjn/pre-trained-lm/roberta-base
model_name=roberta-base
data_path=/wjn/nlp_task_datasets/wikipedia_corpus

# 参数设置
# 在--do_train的下一个添加 --pre_train_from_scratch  gradient_accumulation_steps=16 save_steps=1000
# data_path=/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_pretrain_kgicl_gpt bz:4, gradient_accumulation_steps=2, save_steps:1000

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 6013 nlp_runner.py \
--model_name_or_path=$path \
--data_dir=$data_path \
--train_file=$data_path/total_pretrain_data_10000.txt \
--max_seq_length=512 \
--output_dir=/wjn/frameworks/HugNLP/output/pretrain/mlm_$model_name/ \
--do_train \
--per_device_train_batch_size=2 \
--per_device_eval_batch_size=2 \
--evaluation_strategy=no \
--save_strategy=steps \
--gradient_accumulation_steps=8 \
--learning_rate=1e-05 \
--logging_steps=10000000 \
--save_steps=1000 \
--save_total_limit=20 \
--num_train_epochs=5 \
--report_to=none \
--task_name=mlm_text_line \
--task_type=mlm \
--model_type=roberta \
--exp_name=mlm \
--warmup_steps=40 \
--ignore_data_skip \
--remove_unused_columns=False \
--fp16 \
--max_eval_samples=30000 \
--cache_dir=/wjn/.cache \
--dataloader_num_workers=1 \
--overwrite_output_dir
```
**超参数设置**：
机器配置：4卡V100（32G）

| 语料数量 | 10k（1万） | 100k（10万） | 1M（100万） | 10M（1000万） |
| --- | --- | --- | --- | --- |
| batch_size |  |  |  |  |
| gradient_accumulation_steps |  |  |  |  |
| learning_rate |  |  |  |  |
| mlm_probability |  |  |  |  |

**运行效果**：
![image.png](https://cdn.nlark.com/yuque/0/2023/png/12897066/1676618058372-8063c1cc-20c6-4b89-bdc0-db34ab288c19.png#averageHue=%230b0b0b&clientId=u22b51b75-baf9-4&from=paste&height=321&id=u593f9586&name=image.png&originHeight=642&originWidth=1196&originalType=binary&ratio=2&rotation=0&showTitle=false&size=125161&status=done&style=none&taskId=u1a256fe7-3eb7-41cb-b701-1af74cbb125&title=&width=598)
