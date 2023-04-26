# 每个task重复执行4～5次，每次执行完模型会自动生成pseudo data到相应task的数据集存放地址，接下来的执行会重新还在train和pseudo data
# 每次执行时，需要手动递增修改epoch，根据后面的注释设置相应的epoch
# 全部跑完后，记得保存最终的结果，并删除output里面的地址，下一次训练时避免接着之前训练的checkpoint
#### pre-trained lm path
path=/wjn/pre-trained-lm/chinese-macbert-large
#path=/wjn/pre-trained-lm/chinese-roberta-wwm-ext-large
#path=/wjn/pre-trained-lm/chinese-pert-large
#path=/wjn/pre-trained-lm/simcse-chinese-roberta-wwm-ext
#path=/wjn/pre-trained-lm/erlangshen-10m-sop/Erlangshen-MegatronBert-1.3B
#path=/wjn/pre-trained-lm/erlangshen-5m-cp/Erlangshen-MegatronBert-1.3B



#### task data path
#data_path=/wjn/competition/clue/datasets/ # 203
data_path=/wjn/clue/datasets/CLUEdatasets/ # A100


#### task name
task_name_=clue

#### clue task
#clue_task=wsc
#clue_task=afqmc
clue_task=ocnli
#clue_task=csl
#clue_task=iflytek # mrc不适合类别数量太多的
#clue_task=tnews
#clue_task=tnews_efl
#clue_task=cmnlic/cmnli_public

#clue_task=text_similarity


#### task_type
#task_type_=fusion_siamese
#task_type_=autocls
task_type_=classification

#### inference model path
#path=/wjn/nlp_runner/outputs/clue/erlangshen-5m-cp/$clue_task/Erlangshen-MegatronBert-1.3B/Erlangshen-MegatronBert-1.3B

rm -rf /root/.cache/huggingface/datasets/clue/cache_"$task_name_"_"$clue_task"_train.arrow

if [ "$clue_task" = "wsc" ]; then
  len=128
  bz=4 # 8
  epoch=10 # 10, 20, 30, 40, 50
  eval_step=25
  wr_step=50
  lr=2e-05
  pseudo_threshold=1.0

elif [ "$clue_task" = "afqmc" ]; then
  # continual pre-train
#  path=/wjn/nlp_runner/outputs/clue/erlangshen-5m-cp/text_similarity/Erlangshen-MegatronBert-1.3B/Erlangshen-MegatronBert-1.3B
  len=256
  bz=8
  epoch=2 # 1, 2, 3, 4, 5
  eval_step=200
  wr_step=200
  lr=2e-05
  pseudo_threshold=0.995

elif [ "$clue_task" = "ocnli" ]; then
  len=128
  bz=8 # 8
  epoch=4 # 2, 4, 6, 8, 10
  eval_step=500
  wr_step=500
  lr=3e-05
  pseudo_threshold=0.95

elif [ "$clue_task" = "csl" ]; then
  len=256
  bz=8
  epoch=4 # 4, 8, 12, 16, 20
  eval_step=500
  wr_step=1000
  lr=2e-05
  pseudo_threshold=1.0

elif [ "$clue_task" = "iflytek" ]; then
  len=128
  bz=8
  epoch=2 # 2, 4, 6, 8, 10
  eval_step=200
  wr_step=200
  lr=5e-05
  pseudo_threshold=1.0

elif [ "$clue_task" = "tnews" ]; then
  len=128
  bz=8
  epoch=2, 2, 4, 6, 8, 10
  eval_step=200
  wr_step=100
  lr=2e-05
  pseudo_threshold=1.0

elif [ "$clue_task" = "tnews_efl" ]; then
  len=128
  bz=8
  epoch=2
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


export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port=6010 hugnlp_runner.py \
  --model_name_or_path=$path \
  --data_dir=$data_path/$clue_task \
  --output_dir=./outputs/clue/pseudo/$clue_task \
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
  --task_name=$task_name_ \
  --task_type=$task_type_ \
  --model_type=bert \
  --metric_for_best_model=acc \
  --pad_to_max_length=True \
  --remove_unused_columns=False \
  --overwrite_output_dir \
  --fp16 \
  --label_names=labels \
  --keep_predict_labels \
  --user_defined="data_name=$clue_task is_pseudo=True pseudo_threshold=$pseudo_threshold" \
  --do_adv
