# path=/wjn/pre-trained-lm/gpt2
# path=/wjn/pre-trained-lm/gpt2-medium
# path=/wjn/pre-trained-lm/gpt2-large
path=/wjn/pre-trained-lm/gpt2-xl
# model_name=gpt2
# model_name=gpt2-medium
# model_name=gpt2-large
model_name=gpt2-xl

# gpt2/medium/large gradient_accumulation_steps=2, train_batch=4 (4 gpus)  total bz = 2*4*4=32
# large gradient_accumulation_steps=2, train_batch=2 (8 gpus) (8 gpus) total bz = 2*2*8=32

# mep: gradient_accumulation_steps=16, save_steps=200
# kag: gradient_accumulation_steps=2, save_steps=400

# data_path=/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_pretrain_prompt_data_gpt_shuffle # 所有任务打乱混合
# data_path=/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_pretrain_prompt_data_gpt_mep # 只有MEP任务
# data_path=/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_pretrain_prompt_data_gpt_ecg # 只有ECG任务
# data_path=/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_pretrain_prompt_data_gpt_mep_only_train_last
# data_path=/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_pretrain_prompt_data_gpt_ecg_only_train_last
# data_path=/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_nlp_task_data/ # 只有KAG任务

data_path=/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_pretrain_kgicl_gpt # 全部训练任务数据汇总：依次为ECG + MEP + MAG，共30w语料

# 参数设置
# 在--do_train的下一个添加 --pre_train_from_scratch  gradient_accumulation_steps=16 save_steps=1000
# data_path=/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_pretrain_kgicl_gpt bz:4, gradient_accumulation_steps=2, save_steps:1000

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 6013 hugnlp_runner.py \
--model_name_or_path=$path \
--data_dir=$data_path \
--train_file=$data_path/prompt_data.json \
--max_seq_length=1024 \
--output_dir=/wjn/frameworks/HugNLP/output_kgicl/pretrain/incontext_$model_name/ \
--do_train \
--per_device_train_batch_size=2 \
--per_device_eval_batch_size=2 \
--evaluation_strategy=no \
--save_strategy=steps \
--gradient_accumulation_steps=2 \
--learning_rate=1e-05 \
--logging_steps=10000000 \
--save_steps=1000 \
--save_total_limit=20 \
--num_train_epochs=2 \
--report_to=none \
--task_name=causal_lm_incontext \
--task_type=gpt2_causal_lm \
--exp_name=gpt2_causal_lm \
--warmup_steps=400 \
--model_type=gpt2 \
--ignore_data_skip \
--remove_unused_columns=False \
--fp16 \
--max_eval_samples=30000 \
--cache_dir=/wjn/.cache \
--dataloader_num_workers=1 \
--overwrite_output_dir 