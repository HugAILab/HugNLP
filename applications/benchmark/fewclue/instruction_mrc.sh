# path=/wjn/pre-trained-lm/chinese-macbert-large # pre-trained
path=/wjn/frameworks/HugNLP/outputs/zh_instruction/chinese-macbert-large # continual pre-trained
#path=/wjn/pre-trained-lm/chinese_pretrain_mrc_macbert_large
#path=/wjn/pre-trained-lm/chinese-pert-large
#path=/wjn/pre-trained-lm/chinese-pert-large-mrc
# path=/wjn/pre-trained-lm/chinese-roberta-wwm-ext-large # 78
#path=/wjn/pre-trained-lm/chinese_pretrain_mrc_roberta_wwm_ext_large
#path=/wjn/pre-trained-lm/chinesebert-large
#path=/wjn/pre-trained-lm/structbert-large-zh
#path=/wjn/pre-trained-lm/Erlangshen-MegatronBert-1.3B


#### clue task
# clue_task=tnews
#clue_task=iflytek
#clue_task=csl
#clue_task=wsc
# clue_task=eprstmt
#clue_task=csldcp
clue_task=bustm
#clue_task=chid


data_path=/wjn/nlp_task_datasets/FewCLUEdatasets/$clue_task
if [ "$clue_task" = "wsc" ]; then
  data_path=/wjn/nlp_task_datasets/FewCLUEdatasets/cluewsc
fi

export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.launch --nproc_per_node=2 --master_port=6019 hugnlp_runner.py \
  --model_name_or_path=$path \
  --data_dir=$data_path\
  --output_dir=./outputs/fewclue/$clue_task \
  --seed=42 \
  --exp_name=zh-mrc-instruction-wjn \
  --max_seq_length=512 \
  --max_eval_seq_length=512 \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size=4 \
  --per_device_eval_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --evaluation_strategy=steps \
  --learning_rate=1e-05 \
  --num_train_epochs=32 \
  --logging_steps=100000000 \
  --eval_steps=50 \
  --save_steps=50 \
  --save_total_limit=1 \
  --warmup_steps=50 \
  --load_best_model_at_end \
  --report_to=none \
  --task_name=fewclue_instruction \
  --task_type=bert_global_pointer \
  --model_type=bert \
  --metric_for_best_model=macro_f1 \
  --pad_to_max_length=True \
  --remove_unused_columns=False \
  --overwrite_output_dir \
  --fp16 \
  --label_names=short_labels \
  --keep_predict_labels \
  --cache_dir=/wjn/.cache \
  --user_defined="data_name=$clue_task" \
  # --do_adv