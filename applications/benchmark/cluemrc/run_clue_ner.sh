#### pre-trained lm path
#path=/wjn/pre-trained-lm/chinese_pretrain_mrc_roberta_wwm_ext_large
#path=/wjn/pre-trained-lm/chinese_pretrain_mrc_macbert_large
path=/wjn/pre-trained-lm/chinese-macbert-large
#path=/wjn/pre-trained-lm/chinese-pert-large
#path=/wjn/nlp_runner/outputs/mlm/chinese-macbert-large/chinese-macbert-large
#path=/wjn/pre-trained-lm/erlangshen-5m-cp/Erlangshen-MegatronBert-1.3B
#path=/wjn/pre-trained-lm/erlangshen-10m-sop/Erlangshen-MegatronBert-1.3B
#path=./outputs/clue/24-5m-cp/ocnli/chinese-macbert-large/chinese-macbert-large

#model_type=erlangshen-10m-cp
model_type=chinese_pretrain_mrc_macbert_large
#model_type=chinese-macbert-large
#model_type=chinese-pert-large

#### task data path
#data_path=/wjn/competition/clue/datasets/ # 203
data_path=/wjn/clue/datasets/CLUEdatasets/ # A100


#### task name
task_name_=clue_ner

#### clue task
clue_task=clue_ner

#clue_task=text_similarity



#### inference model path
#path=/wjn/nlp_runner/outputs/clue/erlangshen-10m-cp/clue_ner/Erlangshen-MegatronBert-1.3B/Erlangshen-MegatronBert-1.3B
path=/wjn/nlp_runner/outputs/clue/chinese_pretrain_mrc_macbert_large/clue_ner/chinese_pretrain_mrc_macbert_large/chinese_pretrain_mrc_macbert_large
#path=/wjn/nlp_runner/outputs/clue/chinese-macbert-large/clue_ner/chinese-macbert-large/chinese-macbert-large


if [ "$clue_task" = "clue_ner" ]; then
  len=256
  bz=8 # 8
  epoch=5
  eval_step=1000
  wr_step=500
  lr=2e-05
fi


export CUDA_VISIBLE_DEVICES=4,5
python -m torch.distributed.launch --nproc_per_node=2 --master_port=6010 hugnlp_runner.py \
  --model_name_or_path=$path \
  --data_dir=$data_path/$clue_task \
  --output_dir=./outputs/clue/$model_type/$clue_task \
  --seed=42 \
  --exp_name=clue-wjn \
  --max_seq_length=$len \
  --max_eval_seq_length=$len \
  --do_predict \
  --per_device_train_batch_size=$bz \
  --per_device_eval_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --evaluation_strategy=steps \
  --learning_rate=$lr \
  --num_train_epochs=$epoch \
  --logging_steps=100000000 \
  --eval_steps=$eval_step \
  --save_steps=$eval_step \
  --save_total_limit=1 \
  --warmup_steps=$wr_step \
  --load_best_model_at_end \
  --report_to=none \
  --task_name=$task_name_ \
  --task_type=bert_global_pointer \
  --model_type=bert \
  --metric_for_best_model=macro_f1 \
  --pad_to_max_length=True \
  --remove_unused_columns=False \
  --label_names=short_labels \
  --fp16 \
  --keep_predict_labels \
  --user_defined="data_name=$clue_task" \
  --do_adv
