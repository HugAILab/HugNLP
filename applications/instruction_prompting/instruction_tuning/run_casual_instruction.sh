path=/wjn/pre-trained-lm/gpt2
# path=/wjn/pre-trained-lm/gpt2-medium
# path=/wjn/pre-trained-lm/gpt2-large
# path=/wjn/pre-trained-lm/gpt2-xl


model_name=gpt2

data_path=/wjn/nlp_task_datasets/instruction/all/

export CUDA_VISIBLE_DEVICES=0
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 6013 hugnlp_runner.py \
--model_name_or_path=$path \
--data_dir=$data_path \
--max_seq_length=512 \
--output_dir=/wjn/frameworks/HugNLP/output/pretrain/casual_lm_$model_name/ \
--do_train \
--do_eval \
--do_predict \
--per_device_train_batch_size=1 \
--per_device_eval_batch_size=1 \
--evaluation_strategy=steps \
--save_strategy=steps \
--gradient_accumulation_steps=2 \
--learning_rate=1e-05 \
--logging_steps=10000000 \
--save_steps=1000 \
--eval_steps=1000 \
--save_total_limit=2 \
--num_train_epochs=5 \
--report_to=none \
--task_name=causal_instruction \
--task_type=causal_lm \
--model_type=gpt2 \
--exp_name=causal_lm \
--warmup_steps=1000 \
--ignore_data_skip \
--remove_unused_columns=False \
--fp16 \
--cache_dir=/wjn/.cache \
--overwrite_output_dir
