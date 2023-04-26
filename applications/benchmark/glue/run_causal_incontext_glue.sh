#### pre-trained lm path
# path=/wjn/pre-trained-lm/gpt2
path=/wjn/pre-trained-lm/gpt2-xl
MODEL_TYPE=gpt2

data_path=./ # you can ignore it

#### glue task
# glue_task=cola
# glue_task=mnli
# glue_task=mrpc
# glue_task=sst2
# glue_task=qqp
#glue_task=qnli
glue_task=rte
#glue_task=snli

export CUDA_VISIBLE_DEVICES=0
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=6013 hugnlp_runner.py \
  --model_name_or_path=$path \
  --data_dir=$data_path\
  --output_dir=./outputs/glue/$glue_task/incontext \
  --seed=42 \
  --exp_name=gpt2-incontext-cls \
  --max_seq_length=512 \
  --max_eval_seq_length=512 \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size=1 \
  --per_device_eval_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --evaluation_strategy=steps \
  --logging_steps=100000000 \
  --eval_steps=1 \
  --save_steps=1 \
  --save_total_limit=1 \
  --load_best_model_at_end \
  --report_to=none \
  --task_name=glue_instruction \
  --task_type=causal_prompt_cls \
  --model_type=$MODEL_TYPE \
  --metric_for_best_model=macro_f1 \
  --pad_to_max_length=True \
  --remove_unused_columns=False \
  --overwrite_output_dir \
  --label_names=short_labels \
  --keep_predict_labels \
  --cache_dir=/wjn/.cache \
  --user_defined="data_name=$glue_task num_incontext_example=4 l=1 use_calibrate=True" \
  --use_prompt_for_cls
