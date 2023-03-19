#### pre-trained lm path
path=/wjn/pre-trained-lm/chinese-macbert-base/
MODEL_TYPE=bert

DATA_NAME=default-cls

#### task data path (use should change this path)
data_path=/wjn/frameworks/HugNLP/datasets/data_example/cls

TASK_TYPE=head_cls
# TASK_TYPE=masked_prompt_prefix_cls

len=196
bz=4 # 8
epoch=10
eval_step=50
wr_step=10
lr=1e-05


export CUDA_VISIBLE_DEVICES=0,1
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=6014 hugnlp_runner.py \
--model_name_or_path=$path \
--data_dir=$data_path \
--output_dir=./outputs/default/sequence_classification/$DATA_NAME/ \
--seed=42 \
--exp_name=default-cls \
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
--task_name=default_cls \
--task_type=$TASK_TYPE \
--model_type=$MODEL_TYPE \
--metric_for_best_model=acc \
--pad_to_max_length=True \
--remove_unused_columns=False \
--overwrite_output_dir \
--fp16 \
--label_names=labels \
--keep_predict_labels \
--user_defined="label_names=entailment,neutral,contradiction data_name=$DATA_NAME" \
# --use_prompt_for_cls
