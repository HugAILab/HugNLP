path=/wjn/pre-trained-lm/erlangshen-5m-cp/Erlangshen-MegatronBert-1.3B
#path=/wjn/nlp_runner/outputs/clue/erlangshen-5m-cp/chid/Erlangshen-MegatronBert-1.3B/Erlangshen-MegatronBert-1.3B
data_path=/wjn/clue/datasets/CLUEdatasets/

clue_task=chid
task_name_=clue_chid
#task_name_=chid_mlm
if [ "$task_name_" = "clue_chid" ]; then
  task_type_=megatron_multi_choice_tag
  epoch=4
  output_dir_=./outputs/clue/erlangshen-5m-cp/$clue_task
elif [ "$task_name_" = "chid_mlm" ]; then
  task_type_=chid_mlm
  epoch=4
  output_dir_=./outputs/clue/erlangshen-5m-cp/$clue_task/mlm
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=6023 hugnlp_runner.py \
  --model_name_or_path=$path \
  --data_dir=$data_path/$clue_task \
  --output_dir=$output_dir_ \
  --seed=42 \
  --exp_name=clue-wjn \
  --max_seq_length=256 \
  --max_eval_seq_length=256 \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size=4 \
  --per_device_eval_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --evaluation_strategy=steps \
  --save_strategy=steps \
  --learning_rate=2e-05 \
  --num_train_epochs=$epoch \
  --logging_steps=100000000 \
  --eval_steps=5000 \
  --save_steps=5000 \
  --save_total_limit=1 \
  --warmup_steps=1000 \
  --load_best_model_at_end \
  --report_to=none \
  --task_name=$task_name_ \
  --task_type=$task_type_ \
  --model_type=erlangshen \
  --metric_for_best_model=acc \
  --pad_to_max_length=True \
  --remove_unused_columns=False \
  --overwrite_output_dir \
  --fp16 \
  --label_names=labels \
  --keep_predict_labels \
  --preprocessing_num_workers=1 \
  --do_adv
#  --dataloader_num_workers=1
