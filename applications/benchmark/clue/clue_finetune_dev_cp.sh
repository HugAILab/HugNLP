#path=/wjn/pre-trained-lm/chinese-macbert-24-5m-cp/chinese-macbert-large
#path=/wjn/pre-trained-lm/chinese-macbert-28-5m-cp/chinese-macbert-large
#path=/wjn/pre-trained-lm/chinese-macbert-32-5m-cp/chinese-macbert-large
#path=/wjn/pre-trained-lm/chinese-macbert-32-5m-cp-stackbert/chinese-macbert-large/
path=/wjn/pre-trained-lm/erlangshen-5m-cp/Erlangshen-MegatronBert-1.3B

#path=./outputs/clue/24-5m-cp/ocnli/chinese-macbert-large/chinese-macbert-large


#clue_task=wsc
#clue_task=afqmc
#clue_task=ocnli
#clue_task=csl
#clue_task=iflytek # mrc不适合类别数量太多的
clue_task=tnews
#clue_task=cmnlic/cmnli_public

if [ "$clue_task" = "wsc" ]; then
  len=128
  bz=1 # 8
  epoch=50
  eval_step=25
  wr_step=50
  lr=2e-05

elif [ "$clue_task" = "afqmc" ]; then
  len=256
  bz=1
  epoch=5
  eval_step=500
  wr_step=500
  lr=2e-05

elif [ "$clue_task" = "ocnli" ]; then
  len=128
  bz=1 # 8
  epoch=5
  eval_step=500
  wr_step=500
  lr=3e-05

elif [ "$clue_task" = "csl" ]; then
  len=256
  bz=8
  epoch=5
  eval_step=500
  wr_step=500
  lr=1e-05

elif [ "$clue_task" = "iflytek" ]; then
  len=128
  bz=8
  epoch=5
  eval_step=400
  wr_step=500
  lr=5e-05

elif [ "$clue_task" = "tnews" ]; then
  len=128
  bz=1
  epoch=5
  eval_step=500
  wr_step=500
  lr=2e-05
fi


export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 --master_port=6016 hugnlp_runner.py \
  --model_name_or_path=$path \
  --data_dir=/wjn/competition/clue/datasets/$clue_task \
  --output_dir=./outputs/clue/erlangshen-5m-cp/$clue_task \
  --seed=42 \
  --exp_name=clue-wjn \
  --max_seq_length=$len \
  --max_eval_seq_length=$len \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size=$bz \
  --per_device_eval_batch_size=4 \
  --gradient_accumulation_steps=2 \
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
  --task_name=clue \
  --task_type=classification \
  --model_type=bert \
  --metric_for_best_model=macro_f1 \
  --pad_to_max_length=True \
  --remove_unused_columns=False \
  --overwrite_output_dir \
  --fp16 \
  --keep_predict_labels \
  --user_defined="data_name=$clue_task" \
  --sharded_ddp=zero_dp_2 # If use erlangshen model
#  --do_adv