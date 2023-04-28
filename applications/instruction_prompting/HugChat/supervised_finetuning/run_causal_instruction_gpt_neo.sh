# path=/wjn/pre-trained-lm/gpt-neo-1.3B
path=/wjn/pre-trained-lm/gpt-neo-2.7B


model_name=gpt-neo

# the data example is shown in ./datasets/data_example/instruction/
# we provide an instruction-tuning medical trainin data.
# the training data can be downloaded by run 'bash download_example.sh'
# you can merge all data to form a final train.json and dev.json. the test.json is the same as dev.json.
# data_path=/wjn/nlp_task_datasets/instruction/all # 500,000
data_path=/wjn/nlp_task_datasets/instruction/instruction_corpora # 5,000,000 example, 160k group block

# config1: gpt-neo-1.3B ZeRO stage=1, fp16, train_bz=8, gradient_acc=1, num_gpus=4. cost: 31G per gpu.
# config2: gpt-neo-2.7B ZeRO stage=3, fp16, train_bz=8, gradient_acc=1, num_gpus=4. cost: 39G per gpu.

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 6013 hugnlp_runner.py \
--model_name_or_path=$path \
--data_dir=$data_path \
--max_seq_length=512 \
--output_dir=./outputs/instruction/casual_lm_$model_name/ \
--do_train \
--per_device_train_batch_size=8 \
--per_device_eval_batch_size=1 \
--evaluation_strategy=no \
--save_strategy=steps \
--gradient_accumulation_steps=1 \
--learning_rate=2e-05 \
--logging_steps=10000000 \
--save_steps=10000 \
--save_total_limit=10 \
--num_train_epochs=2 \
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
--user_defined="causal_lm_name=$model_name language=en" \
--deepspeed=./deepspeed/ds_config_fp16_z3.json \
--fp16
