# path=/wjn/pre-trained-lm/opt-1.3b
path=facebook/opt-6.7b

model_name=opt

# the data example is shown in ./datasets/data_example/instruction/
# we provide an instruction-tuning medical trainin data.
# the training data can be downloaded by run 'bash download_example.sh'
# you can merge all data to form a final train.json and dev.json. the test.json is the same as dev.json.
# data_path=/wjn/nlp_task_datasets/instruction/all # 500,000
data_path=/wjn/nlp_task_datasets/instruction/instruction_corpora

# config1: ZeRO stage=3, fp16, train_bz=4, gradient_acc=1, num_gpus=8, lora_dim=8， cost gpu 25G

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 6013 hugnlp_runner.py \
--model_name_or_path=$path \
--data_dir=$data_path \
--max_seq_length=512 \
--output_dir=./outputs/instruction/causal_lm_$model_name/ \
--do_train \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=1 \
--evaluation_strategy=no \
--save_strategy=steps \
--gradient_accumulation_steps=1 \
--learning_rate=1e-4 \
--logging_steps=10000000 \
--save_steps=10000 \
--save_total_limit=10 \
--num_train_epochs=2 \
--lora_dim=8 \
--report_to=none \
--task_name=causal_instruction \
--task_type=auto_causal_lm \
--model_type=auto \
--exp_name=causal-instruciton \
--warmup_steps=6000 \
--ignore_data_skip \
--remove_unused_columns=False \
--cache_dir=/wjn/.cache \
--overwrite_output_dir \
--user_defined="causal_lm_name=$model_name stop_token=<|endoftext|> language=en" \
--deepspeed=./deepspeed/ds_config_fp16_z3.json \
--fp16