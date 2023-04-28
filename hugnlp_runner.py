# -*- coding: utf-8 -*-
# @Time    : 2022/08/29 4:09 下午
# @Author  : JianingWang
# @File    : hugnlp_runner.py
# !/usr/bin/env python
# coding=utf-8

"""
This file is the runner of HugNLP.
"""
import math
import os
import time
import torch
import numpy as np
from callback.ema import ExponentialMovingAveragingCallback
from callback.freeze import FreezeCallback
from callback.logger import LoggerCallback
from processors import PROCESSORS
from transformers import CONFIG_MAPPING, AutoConfig, AutoTokenizer, HfArgumentParser
from hugnlp_trainer import HugTrainer, HugSelfTrainer
from transformers.trainer_utils import get_last_checkpoint
from transformers import EarlyStoppingCallback
from config import ModelArguments, DataTrainingArguments, TrainingArguments, SemiSupervisedTrainingArguments
from callback.mlflow import MLflowCallback
from tools.runner_utils.log_util import init_logger
from models import MODEL_CLASSES, TOKENIZER_CLASSES
from models.basic_modules.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters
from evaluators import EVALUATORS_CLASSES
from tools.runner_utils.conifg_extensive import config_extensive
from tools.runner_utils.set_seed import set_seed
import logging

logger = logging.getLogger(__name__)
torch.set_printoptions(precision=3, edgeitems=5, linewidth=160, sci_mode=False)

def print_hello():
    length = 82
    print("+" + "-"*(length - 2) + "+")
    print("|" + " "*(length - 2) + "|")
    print(" " + " "*int((length - 2 - 25)/2) + "🤗 Welcome to use HugNLP!" + " "*int((length - 2 - 25)/2)  + " ")
    print("" + " "*(length) + "")
    print("" + " "*int((length - 2 - 32)/2) + "https://github.com/HugAILab/HugNLP" + " "*int((length - 2 - 33)/2)  + "")
    print("|" + " "*(length - 2) + "|")
    print("+" + "-"*(length - 2) + "+")

def main():
    # See all possible arguments or by passing the --help flag to this script.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, SemiSupervisedTrainingArguments))
    model_args, data_args, training_args, semi_training_args = parser.parse_args_into_dataclasses()

    # Print hello world
    if training_args.local_rank == 0:
        print_hello()

    training_args.output_dir = os.path.join(training_args.output_dir, list(filter(None, model_args.model_name_or_path.split("/")))[-1])
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(training_args.output_dir,
                            f"{model_args.model_name_or_path.split(os.sep)[-1]}-{data_args.task_name}-{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())}.log")
    log_level = training_args.get_process_log_level()
    init_logger(log_file, log_level, training_args.local_rank)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if training_args.local_rank == 0:
        logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len([i for i in os.listdir(training_args.output_dir) if not i.endswith("log")]) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                             "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Obtain tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        # "do_lower_case": model_args.do_lower_case #根据model 的tokenizer自己配置
    }
    tokenizer_class = TOKENIZER_CLASSES.get(model_args.model_type, AutoTokenizer)
    if model_args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # Build processors
    if data_args.task_name in PROCESSORS:
        processor = PROCESSORS[data_args.task_name](data_args, training_args, model_args, tokenizer=tokenizer)
    else:
        raise ValueError("Unknown task name {}, please check in processor map.".format(data_args.task_name))

    # Load pretrained model and tokenizer
    # The .from_pretrained methods guarantee that only one local process can concurrently download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "finetuning_task": data_args.task_name
    }

    # Add num_labels
    if hasattr(processor, "labels"):
        config_kwargs["num_labels"] = len(processor.labels)

    # Set configure
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
        if "longformer" in model_args.model_name_or_path:
            config.sep_token_id = 102
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # Add label mapping if use prompt-tuning
    if model_args.use_prompt_for_cls:
        assert hasattr(processor, "label_word_list"), "If you use prompt, you must design label_word_list in processor."
        config.label_word_list = processor.label_word_list

    # Add other config
    config = config_extensive(config, model_args)
    processor.set_config(config)

    # Set model
    model_class = MODEL_CLASSES[data_args.task_type]
    if type(model_class) == dict:
        model_class = model_class[model_args.model_type]

    if training_args.pre_train_from_scratch:
        logger.info("Training new model from scratch")
        model = model_class(config)
    else:
        logger.info("Continual Tuning a Pre-trained Model")
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            # ignore_mismatched_sizes=True
        )

    # Resize token embeddings
    try:
        model.resize_token_embeddings(len(tokenizer))
    except:
        print("Fail to resize token embeddings.")

    if model_args.lora_dim > 0 and training_args.deepspeed is not None:
        from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model
        # model = prepare_model_for_int8_training(model) # INT8 量化
        # load lora
        logger.info("You have set LORA parameter-efficient learning.")
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)

    # if model_args.lora_dim > 0 and training_args.deepspeed is not None:
    #     # 插入lora参数
    #     model = convert_linear_layer_to_lora(
    #         model, model_args.lora_module_name,
    #         model_args.lora_dim)
    #     # 只对lora参数进行训练
    #     if model_args.only_optimize_lora:
    #         model = only_optimize_lora_parameters(model)

    train_dataset, eval_dataset, test_dataset, unlabeled_dataset = None, None, None, None

    # Obtain tokenized data
    tokenized_datasets = processor.get_tokenized_datasets()
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in tokenized_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = tokenized_datasets["test"]
        if data_args.max_predict_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_predict_samples))

    if semi_training_args.use_semi:
        assert "unlabeled_data" in tokenized_datasets.features, "If you choose semi-supervised training, you must define unlabeled data."
        unlabeled_dataset = tokenized_datasets["unlabeled_dataset"]
        if semi_training_args.unlabeled_data_num is not None:
            unlabeled_dataset = unlabeled_dataset.select(range(semi_training_args.unlabeled_data_num))

    # Set evaluator
    assert data_args.task_type in EVALUATORS_CLASSES, "You must define an evaluator for '{}'".format(data_args.task_type)
    evaluator_class = EVALUATORS_CLASSES[data_args.task_type]
    evaluator = evaluator_class(
        model_args=model_args,
        data_args=data_args,
        model=model,
        training_args=training_args,
        processor=processor,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
    )

    # Set data collator
    data_collator = processor.get_data_collator()
    if hasattr(processor, "compute_metrics"):
        # first to choose defined "compute_metrics" in processor.
        compute_metrics = processor.compute_metrics
    elif hasattr(evaluator, "default_compute_metrics"):
        # second to choose defined "default_compute_metrics" in evaluator.
        compute_metrics = evaluator.default_compute_metrics
    else:
        # set None, in this case, evaluating in training time will not calculate metric.
        compute_metrics = None

    # Obtain tracking
    callbacks = [LoggerCallback]
    if data_args.mlflow_location or data_args.tracking_uri:
        mlflow_callback = MLflowCallback(model_args, data_args, training_args)
        callbacks.append(mlflow_callback)

    if model_args.early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping_patience))

    if model_args.freeze_epochs:
        callbacks.append(FreezeCallback(freeze_epochs=model_args.freeze_epochs, freeze_keyword=model_args.freeze_keyword))

    if model_args.ema:
        callbacks.append(ExponentialMovingAveragingCallback(model_args.ema_decay))

    if training_args.do_predict_during_train:
        from callback.evaluate import DoPredictDuringTraining
        callbacks.append(DoPredictDuringTraining(test_dataset, processor))

    # Obtain trainer
    if not semi_training_args.use_semi:
        # traditional trainer
        trainer = HugTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks
        )
    else:
        # self-trainer
        trainer = HugSelfTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks
        )


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Update trainer state
    evaluator.reset_trainer(trainer if not semi_training_args.use_semi else trainer.student_trainer)
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        evaluator.evaluate()

    # Prediction
    if training_args.do_predict and not training_args.do_predict_during_train:
        logger.info("*** Predict ***")
        evaluator.predict()

if __name__ == "__main__":
    main()
