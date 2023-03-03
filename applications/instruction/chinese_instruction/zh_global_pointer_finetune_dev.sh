path=/wjn/pre-trained-lm/chinese-macbert-large # 89
#path=/wjn/pre-trained-lm/chinese_pretrain_mrc_macbert_large
#path=/wjn/pre-trained-lm/chinese-pert-large
#path=/wjn/pre-trained-lm/chinese-pert-large-mrc
# path=/wjn/pre-trained-lm/chinese-roberta-wwm-ext-large # 78
#path=/wjn/pre-trained-lm/chinese_pretrain_mrc_roberta_wwm_ext_large
#path=/wjn/pre-trained-lm/chinesebert-large
#path=/wjn/pre-trained-lm/structbert-large-zh
#path=/wjn/pre-trained-lm/Erlangshen-MegatronBert-1.3B

# data_path=/wjn/nlp_task_datasets/zh_instruction
data_path=/wjn/nlp_task_datasets/chinese_datahub

export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 --master_port=6019 hugnlp_runner.py \
  --model_name_or_path=$path \
  --data_dir=$data_path\
  --output_dir=./outputs/zh_instruction \
  --seed=42 \
  --exp_name=zh-mrc-instruction-wjn \
  --max_seq_length=512 \
  --max_eval_seq_length=512 \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size=32 \
  --per_device_eval_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --evaluation_strategy=steps \
  --learning_rate=2e-05 \
  --num_train_epochs=3 \
  --logging_steps=100000000 \
  --eval_steps=500 \
  --save_steps=500 \
  --save_total_limit=1 \
  --warmup_steps=200 \
  --load_best_model_at_end \
  --report_to=none \
  --task_name=zh_mrc_instruction \
  --task_type=bert_global_pointer \
  --model_type=bert \
  --metric_for_best_model=macro_f1 \
  --pad_to_max_length=True \
  --remove_unused_columns=False \
  --overwrite_output_dir \
  --fp16 \
  --label_names=short_labels \
  --keep_predict_labels \
  --cache_dir=/wjn/.cache
  # --do_adv
