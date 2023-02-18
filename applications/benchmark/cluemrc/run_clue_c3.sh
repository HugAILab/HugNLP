path=/wjn/pre-trained-lm/erlangshen-5m-cp/Erlangshen-MegatronBert-1.3B
#path=/wjn/nlp_runner/outputs/clue/erlangshen-5m-cp/c3/Erlangshen-MegatronBert-1.3B/Erlangshen-MegatronBert-1.3B
data_path=/wjn/clue/datasets/CLUEdatasets/
clue_task=c3

export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=6017 hugnlp_runner.py \
  --model_name_or_path=$path \
  --data_dir=$data_path/$clue_task \
  --output_dir=./outputs/clue/erlangshen-5m-cp/$clue_task \
  --seed=42 \
  --exp_name=clue-wjn \
  --max_seq_length=512 \
  --max_eval_seq_length=512 \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size=1 \
  --per_device_eval_batch_size=16 \
  --gradient_accumulation_steps=2 \
  --evaluation_strategy=steps \
  --save_strategy=steps \
  --learning_rate=2e-05 \
  --num_train_epochs=10 \
  --logging_steps=100000000 \
  --eval_steps=200 \
  --save_steps=200 \
  --save_total_limit=1 \
  --warmup_steps=50 \
  --load_best_model_at_end \
  --report_to=none \
  --task_name=c3 \
  --task_type=multi_choice_megatron \
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
