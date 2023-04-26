#### pre-trained lm path
path=/wjn/pre-trained-lm/bert-base-uncased
# path=/wjn/pre-trained-lm/roberta-base

MODEL_TYPE=bert
# MODEL_TYPE=roberta

data_path=./ # you can ignore it

#### task name
TASK_NAME=glue

#### glue task
#glue_task=cola
# glue_task=mnli
#glue_task=mnli-mm
# glue_task=mrpc
#glue_task=sst-2
#glue_task=sts-b
# glue_task=qqp
#glue_task=qnli
glue_task=rte
#glue_task=wnli
#glue_task=snli
#glue_task=mr
#glue_task=sst-5
#glue_task=subj
#glue_task=trec
#glue_task=mpqa

#### task_type
# TASK_TYPE=autocls
# TASK_TYPE=classification
# TASK_TYPE=head_cls
TASK_TYPE=head_prefix_cls
# TASK_TYPE=masked_prompt_prefix_cls

### hyper-parameter
len=128
bz=8
epoch=100
eval_step=50
wr_step=50
lr=1e-05



# if [ "$clue_task" = "wsc" ]; then
#   len=196
#   bz=4 # 8
#   epoch=10
#   eval_step=50
#   wr_step=50
#   lr=3e-05
#   TASK_TYPE=clue_wsc
  # rm -rf /root/.cache/huggingface/datasets/clue/cache_"$task_name_"_"$clue_task"_validation.arrow
  # rm -rf /root/.cache/huggingface/datasets/clue/cache_"$task_name_"_"$clue_task"_test.arrow

# elif [ "$clue_task" = "ocnli" ]; then
#   len=128
#   bz=8 # 8
#   epoch=4
#   eval_step=500
#   wr_step=500
#   lr=3e-05
# fi

#rm -rf /root/.cache/huggingface/datasets/clue/cache_"$task_name_"_"$clue_task"_train.arrow


export CUDA_VISIBLE_DEVICES=0,1
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=6014 hugnlp_runner.py \
  --model_name_or_path=$path \
  --data_dir=$data_path \
  --output_dir=./outputs/glue/$glue_task \
  --seed=42 \
  --exp_name=glue-wjn \
  --max_seq_length=$len \
  --max_eval_seq_length=$len \
  --do_train \
  --do_eval \
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
  --model_type=$MODEL_TYPE \
  --metric_for_best_model=acc \
  --pad_to_max_length=True \
  --remove_unused_columns=False \
  --overwrite_output_dir \
  --label_names=labels \
  --keep_predict_labels \
  --user_defined="data_name=$glue_task" \
  # --use_prompt_for_cls \
  # --pre_seq_len=1 \
  # --use_freezing
  # --do_adv
