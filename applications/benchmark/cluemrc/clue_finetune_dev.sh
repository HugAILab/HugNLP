#path=/wjn/competition/clue/nlp_runner/outputs/merge/saved_macbert/chinese-macbert-large
path=/wjn/competition/clue/nlp_runner/outputs/merge/saved_macbert_mrc/chinese_pretrain_mrc_macbert_large
#path=/wjn/competition/clue/nlp_runner/outputs/merge/saved_pert/chinese-pert-large
#path=/wjn/competition/clue/nlp_runner/outputs/merge/saved_pert_mrc/chinese-pert-large-mrc
#path=/wjn/competition/clue/nlp_runner/outputs/merge/saved_roberta/chinese-roberta-wwm-ext-large
#path=/wjn/competition/clue/nlp_runner/outputs/merge/saved_roberta_mrc/chinese_pretrain_mrc_roberta_wwm_ext_large
#path=/wjn/pre-trained-lm/chinese_pretrain_mrc_macbert_large


#clue_task=wsc
#clue_task=afqmc
clue_task=ocnli
#clue_task=csl
#clue_task=iflytek # mrc不适合类别数量太多的
#clue_task=tnews
#clue_task=cmnli/cmnli_public

if [ "$clue_task" = "wsc" ]; then
  len=128
  bz=12
  epoch=50
  eval_step=25
  wr_step=50
  lr=2e-05

elif [ "$clue_task" = "afqmc" ]; then
  len=256
  bz=12
  epoch=5
  eval_step=500
  wr_step=500
  lr=2e-05

elif [ "$clue_task" = "ocnli" ]; then
  len=128
  bz=12
  epoch=5
  eval_step=500
  wr_step=500
  lr=3e-05

elif [ "$clue_task" = "csl" ]; then
  len=512
  bz=8
  epoch=5
  eval_step=500
  wr_step=500
  lr=1e-05

elif [ "$clue_task" = "iflytek" ]; then
  len=512
  bz=8
  epoch=5
  eval_step=400
  wr_step=500
  lr=5e-05
fi
#rm -rf /root/.cache/huggingface/datasets/clue_mrc_style/cache_clue_mrc_style_mrc_style_train.arrow
#rm -rf /root/.cache/huggingface/datasets/clue_mrc_style/cache_clue_mrc_style_mrc_style_validation.arrow
#rm -rf /root/.cache/huggingface/datasets/clue_mrc_style/cache_clue_mrc_style_mrc_style_test.arrow

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 --master_port=6010 hugnlp_runner.py \
  --model_name_or_path=$path \
  --data_dir=/wjn/competition/clue/datasets/$clue_task \
  --output_dir=./outputs/clue/$clue_task \
  --seed=42 \
  --exp_name=cpic-large-wjn \
  --max_seq_length=$len \
  --max_eval_seq_length=$len \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size=$bz \
  --per_device_eval_batch_size=16 \
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
  --task_name=clue_mrc_style \
  --task_type=global_pointer \
  --model_type=bert \
  --metric_for_best_model=macro_f1 \
  --pad_to_max_length=True \
  --remove_unused_columns=False \
  --overwrite_output_dir \
  --fp16 \
  --label_names=short_labels \
  --keep_predict_labels \
  --user_defined="data_name=$clue_task"
#  --do_adv
