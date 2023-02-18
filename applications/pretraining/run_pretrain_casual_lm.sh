path=/wjn/pre-trained-lm/gpt2
# path=/wjn/pre-trained-lm/gpt2-medium
# path=/wjn/pre-trained-lm/gpt2-large
# path=/wjn/pre-trained-lm/gpt2-xl


model_name=gpt2

data_path=/wjn/nlp_task_datasets/wikipedia_corpus

# 参数设置
# 在--do_train的下一个添加 --pre_train_from_scratch  gradient_accumulation_steps=16 save_steps=1000
# data_path=/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_pretrain_kgicl_gpt bz:4, gradient_accumulation_steps=2, save_steps:1000

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 6013 hugnlp_runner.py \
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