#### pre-trained lm path
path=/code/cn/CodePrompt/data/huggingface_models/graphcodebert-base/
MODEL_TYPE=graphcodebert

#### task data path (use should change this path)
data_path=/code/cn/HugNLP/datasets/data_example/clone/

TASK_TYPE=code_cls
# TASK_TYPE=masked_prompt_prefix_cls

len=512
bz=4 # 8
epoch=10
eval_step=50
wr_step=10
lr=1e-05


export CUDA_VISIBLE_DEVICES=0,1
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=6014 hugnlp_runner.py \
--model_name_or_path=$path \
--data_dir=$data_path \
--output_dir=./outputs/code/clone_classification_graphcodebert\
--seed=1234 \
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
--task_name=code_clone \
--task_type=$TASK_TYPE \
--model_type=$MODEL_TYPE \
--metric_for_best_model=acc \
--pad_to_max_length=True \
--remove_unused_columns=False \
--overwrite_output_dir \
--label_names=labels \
--keep_predict_labels \
--user_defined="label_names=0,1 max_target_length=512" \
