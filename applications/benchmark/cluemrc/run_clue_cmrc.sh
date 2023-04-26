path=/wjn/pre-trained-lm/chinese-macbert-large
# path=/wjn/pre-trained-lm/erlangshen-5m-cp/Erlangshen-MegatronBert-1.3B
#path=/wjn/pre-trained-lm/erlangshen-10m-sop/Erlangshen-MegatronBert-1.3B
#path=/wjn/nlp_runner/outputs/clue/erlangshen-5m-cp/cmrc/Erlangshen-MegatronBert-1.3B/Erlangshen-MegatronBert-1.3B
data_path=/wjn/nlp_task_datasets/CLUEdatasets
clue_task=cmrc

export CUDA_VISIBLE_DEVICES=4,5
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=6013 hugnlp_runner.py \
  --model_name_or_path=$path \
  --data_dir=$data_path/$clue_task \
  --output_dir=./outputs/clue/$clue_task \
  --seed=42 \
  --exp_name=clue-wjn \
  --max_seq_length=512 \
  --max_eval_seq_length=512 \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size=16 \
  --per_device_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
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
  --task_name=cmrc18_global_pointer \
  --task_type=global_pointer \
  --model_type=bert \
  --metric_for_best_model=f1 \
  --pad_to_max_length=True \
  --remove_unused_columns=False \
  --overwrite_output_dir \
  --fp16 \
  --label_names=short_labels \
  --keep_predict_labels \
  --preprocessing_num_workers=1 \
  --do_adv
#  --dataloader_num_workers=1
