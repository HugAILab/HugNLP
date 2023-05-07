path=/wjn/pre-trained-lm/gpt2

model_name=gpt2

data_path=/wjn/nlp_task_datasets/rlhf_preference # consists of preference_train.json, preference_dev.json, preference_test.json


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 6013 hugnlp_runner.py \
--model_name_or_path=$path \
--data_dir=$data_path \
--max_seq_length=512 \
--output_dir=./outputs/rlhf/$model_name/ \
--do_train \
--do_eval \
--do_predict \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=1 \
--evaluation_strategy=steps \
--save_strategy=steps \
--gradient_accumulation_steps=1 \
--learning_rate=1e-05 \
--logging_steps=10000000 \
--eval_steps=3000 \
--save_steps=3000 \
--save_total_limit=10 \
--num_train_epochs=3 \
--report_to=none \
--task_name=pairwise_reward \
--task_type=rl_reward \
--model_type=gpt2 \
--exp_name=preference_reward \
--warmup_steps=6000 \
--load_best_model_at_end \
--metric_for_best_model=acc \
--ignore_data_skip \
--remove_unused_columns=False \
--cache_dir=/wjn/.cache \
--overwrite_output_dir \
# --deepspeed=./deepspeed/ds_config_fp16_z1.json \
# --fp16
