#### pre-trained lm path
path=/wjn/pre-trained-lm/chinese-macbert-large
# path=/wjn/pre-trained-lm/chinese-roberta-wwm-ext-large
#path=/wjn/pre-trained-lm/chinese-pert-large
#path=/wjn/pre-trained-lm/simcse-chinese-roberta-wwm-ext


#### task data path
#data_path=/wjn/competition/clue/datasets/ # 203
# data_path=/wjn/clue/datasets/CLUEdatasets/ # A100
data_path=/wjn/nlp_task_datasets/CLUEdatasets # nas A100


#### task name
TASK_NAME=clue

#### clue task
#clue_task=wsc
# clue_task=afqmc
#clue_task=ocnli
# clue_task=csl
#clue_task=csl_efl
#clue_task=iflytek # mrc不适合类别数量太多的
clue_task=tnews
#clue_task=tnews_efl
#clue_task=cmnlic/cmnli_public
# clue_task=qbqtc

#clue_task=text_similarity


#### task_type
#task_type_=fusion_siamese
#task_type_=autocls
# TASK_TYPE=classification
# TASK_TYPE=head_cls
TASK_TYPE=head_prefix_cls

#### inference model path
#path=/wjn/nlp_runner/outputs/clue/ocnli/chinese-macbert-large/chinese-macbert-large/

if [ "$clue_task" = "wsc" ]; then
  len=196
  bz=4 # 8
  epoch=10
  eval_step=50
  wr_step=50
  lr=3e-05
  TASK_TYPE=clue_wsc
  rm -rf /root/.cache/huggingface/datasets/clue/cache_"$task_name_"_"$clue_task"_validation.arrow
  rm -rf /root/.cache/huggingface/datasets/clue/cache_"$task_name_"_"$clue_task"_test.arrow

elif [ "$clue_task" = "afqmc" ]; then
  # continual pre-train
#  path=/wjn/nlp_runner/outputs/clue/erlangshen-5m-cp/text_similarity/Erlangshen-MegatronBert-1.3B/Erlangshen-MegatronBert-1.3B
  len=256
  bz=8
  epoch=2
  eval_step=200
  wr_step=200
  lr=2e-05

elif [ "$clue_task" = "ocnli" ]; then
  len=128
  bz=8 # 8
  epoch=4
  eval_step=500
  wr_step=500
  lr=3e-05

elif [ "$clue_task" = "csl" ]; then
#  path=/wjn/nlp_runner/outputs/clue/csl/chinese-macbert-large/chinese-macbert-large
  len=256
  bz=8
  epoch=20
  eval_step=500
  wr_step=1000
  lr=2e-05

elif [ "$clue_task" = "csl_efl" ]; then
#  path=/wjn/nlp_runner/outputs/clue/csl_efl/chinese-macbert-large/chinese-macbert-large
  len=256
  bz=8
  epoch=20
  eval_step=500
  wr_step=1000
  lr=2e-05
  task_name_=clue_csl_efl

elif [ "$clue_task" = "iflytek" ]; then
  len=128
  bz=8
  epoch=15
  eval_step=200
  wr_step=200
  lr=5e-05

elif [ "$clue_task" = "tnews" ]; then
  len=128
  bz=8
  epoch=10
  eval_step=200
  wr_step=100
  lr=2e-05

elif [ "$clue_task" = "qbqtc" ]; then
  len=64
  bz=16
  epoch=7
  eval_step=200
  wr_step=100
  lr=2e-05

elif [ "$clue_task" = "tnews_efl" ]; then
  len=128
  bz=8
  epoch=15
  eval_step=200
  wr_step=100
  lr=2e-05
  task_name_=clue_tnews_efl

elif [ "$clue_task" = "text_similarity" ]; then
  len=128
  bz=16
  epoch=2
  eval_step=500
  wr_step=200
  lr=2e-05
fi

#rm -rf /root/.cache/huggingface/datasets/clue/cache_"$task_name_"_"$clue_task"_train.arrow


export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 --master_port=6014 hugnlp_runner.py \
  --model_name_or_path=$path \
  --data_dir=$data_path/$clue_task \
  --output_dir=./outputs/clue/$clue_task \
  --seed=42 \
  --exp_name=clue-wjn \
  --max_seq_length=$len \
  --max_eval_seq_length=$len \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size=$bz \
  --per_device_eval_batch_size=4 \
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
  --task_name=$TASK_NAME \
  --task_type=$TASK_TYPE \
  --model_type=bert \
  --metric_for_best_model=acc \
  --pad_to_max_length=True \
  --remove_unused_columns=False \
  --overwrite_output_dir \
  --fp16 \
  --label_names=labels \
  --keep_predict_labels \
  --user_defined="data_name=$clue_task" \
  --pre_seq_len=4 \
  --use_freezing
  # --do_adv