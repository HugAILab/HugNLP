#### pre-trained lm path
path=/wjn/pre-trained-lm/bert-base-uncased/
MODEL_TYPE=bert

DATA_NAME=default-labeling

#### task data path (user should change this path)
data_path=./datasets/data_example/ner

# TASK_TYPE=auto_token_cls
# TASK_TYPE=head_softmax_token_cls
TASK_TYPE=head_crf_token_cls

len=64
bz=4 # 8
epoch=100
eval_step=50
wr_step=10
lr=1e-05


export CUDA_VISIBLE_DEVICES=0,1
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=6014 hugnlp_runner.py \
--model_name_or_path=$path \
--data_dir=$data_path \
--output_dir=./outputs/default/sequence_labeling/$DATA_NAME/ \
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
--task_name=default_labeling \
--task_type=$TASK_TYPE \
--model_type=$MODEL_TYPE \
--metric_for_best_model=acc \
--pad_to_max_length=True \
--remove_unused_columns=False \
--overwrite_output_dir \
--label_names=labels \
--keep_predict_labels \
--user_defined="data_name=user-define"
