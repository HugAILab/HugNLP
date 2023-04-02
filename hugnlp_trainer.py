# -*- coding: utf-8 -*-
# @Time    : 2022/1/7 3:07 下午
# @Author  : JianingWang
# @File    : HugTrainer

import os
import torch
from torch import nn
import datasets
from datasets import Dataset
from processors.dataset import DatasetK
from torch.utils.data import RandomSampler, DistributedSampler
from typing import Dict, Union, Any, Optional, Callable, List, Tuple, Iterator, OrderedDict
from transformers import PreTrainedModel, DataCollator, PreTrainedTokenizerBase, EvalPrediction, TrainerCallback
from transformers.trainer_pt_utils import DistributedSamplerWithLoop, get_length_grouped_indices
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler as DistributedLengthGroupedSamplerOri
from transformers.trainer_pt_utils import LengthGroupedSampler as LengthGroupedSamplerOri
# from transformers.trainer_utils import has_length
from transformers.training_args import ParallelMode
from transformers.trainer import Trainer, _is_torch_generator_available
from transformers.trainer_utils import denumpify_detensorize, TrainOutput
from config import TrainingArguments

from transformers.file_utils import is_datasets_available
from models.adversarial import FGM

from tools.model_utils.uncertainty import sample_by_bald_class_easiness
from tools.runner_utils.log_util import logging
logger = logging.getLogger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"

class LengthGroupedSampler(LengthGroupedSamplerOri):
    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator, mega_batch_mult=256)
        return iter(indices)


class DistributedLengthGroupedSampler(DistributedLengthGroupedSamplerOri):
    def __iter__(self) -> Iterator:
        # Deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=g, mega_batch_mult=400)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[: (self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


"""
Trainer for running HugNLP
"""
class HugTrainer(Trainer):
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super(HugTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers)
        self.metric_for_best_model = self.args.metric_for_best_model
        if self.args.do_adv:
            self.fgm = FGM(self.model)
        for callback in callbacks:
            callback.trainer = self
        self.best_metrics = OrderedDict({
            "best_epoch": 0,
            f"best_eval_{self.metric_for_best_model}": 0,
        })
        self.global_step_ = 0

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model"s documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        self.global_step_ += 1
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1 or len(loss.shape) > 0:
            # 如果是多GPU，或者当前的loss是一个tensor列表
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.global_step_ % 10 == 0:
            print("[step={}, loss={}]".format(self.global_step_, loss))

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()

        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        # 对抗训练
        if self.args.do_adv:
            self.fgm.attack()
            with self.autocast_smart_context_manager():
                loss_adv = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss_adv = loss_adv.mean()
            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                loss_adv = loss_adv / self.args.gradient_accumulation_steps
            if self.do_grad_scaling:
                self.scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()
            self.fgm.restore()  # 恢复embedding参数

        return loss.detach()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # if not has_length(self.train_dataset):
        #     return None

        generator = None
        if self.args.world_size <= 1 and _is_torch_generator_available:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=self.args.seed,
                )

        else:
            if self.args.world_size <= 1:
                if _is_torch_generator_available:
                    return RandomSampler(self.train_dataset, generator=generator)
                return RandomSampler(self.train_dataset)
            elif (
                    self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                    and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=self.args.seed,
                )


"""
Self-trainer for self-training HugNLP
"""
class HugSelfTrainer(object):
    def __init__(
        self,
        teacher_base_model: torch.nn.Module,
        student_base_model: torch.nn.Module,
        training_args,
        semi_training_args,
        train_dataset: Optional[Dataset]=None,
        unlabeled_dataset: Optional[Dataset]=None,
        eval_dataset=None,
        compute_metrics=None,
        tokenizer=None,
        teacher_data_collator=None,
        student_data_collator=None,
        task_type="cls",
        num_classes=0,
    ) -> None:

        logger.info("This is a Self-trainer.")

        self.teacher_base_model = teacher_base_model
        self.student_base_model = student_base_model
        self.training_args = training_args
        self.metric_for_best_model = self.training_args.metric_for_best_model
        self.semi_training_args = semi_training_args
        self.train_dataset = train_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.tokenizer = tokenizer
        self.teacher_data_collator = teacher_data_collator
        self.student_data_collator = student_data_collator
        self.task_type = task_type
        self.num_classes = num_classes

        # self.set_teacher_trainer()
        # self.set_student_trainer()
        self.training_args.per_device_train_batch_size = self.semi_training_args.unlabeled_data_batch_size
        self.teacher_training_epoch = self.semi_training_args.teacher_training_epoch # 最初teacher模型在labeled data上训练的epoch数
        self.teacher_tuning_epoch = self.semi_training_args.teacher_tuning_epoch # 每一轮Self-training时，teacher模型继续在labeled data上tune的epoch数
        self.student_training_epoch = self.semi_training_args.student_training_epoch # 每一轮Self-training时，student模型在pseudo-labeled data上训练的epoch数
        self.self_training_epoch = self.semi_training_args.self_training_epoch # Self-training迭代数
        self.unlabeled_data_num = self.semi_training_args.unlabeled_data_num # self-training每轮迭代时，首先挑选一部分用于计算MC dropout uncertainty。-1表示全部计算uncertainty
        self.pseudo_sample_num_or_ratio = self.semi_training_args.pseudo_sample_num_or_ratio # MC dropout后，从所有计算过uncertainty的unlabeled data上采样的样本比例/数量
        self.student_learning_rate = self.semi_training_args.student_learning_rate
        self.output_dir = self.training_args.output_dir

    def get_teacher_trainer(
        self,
        base_model: torch.nn.Module,
        num_train_epochs: int,
        output_dir: str = None,
        ):
        training_args = self.training_args
        training_args.num_train_epochs = num_train_epochs
        if output_dir is not None:
            training_args.output_dir = output_dir
        # 初始化Teacher训练器
        teacher_trainer = HugTrainer(
            model=base_model,
            args=training_args,
            train_dataset=self.train_dataset if self.training_args.do_train else None,
            eval_dataset=self.eval_dataset if self.training_args.do_eval else None,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=self.teacher_data_collator,
        )
        return teacher_trainer


    def get_student_trainer(
        self,
        base_model: torch.nn.Module,
        num_train_epochs: int,
        student_learning_rate: float,
        pseudo_labeled_dataset: Optional[Dataset] = None,
        output_dir: str = None,
        ):
        training_args = self.training_args
        training_args.num_train_epochs = num_train_epochs
        training_args.learning_rate = student_learning_rate
        if output_dir is not None:
            training_args.output_dir = output_dir
        # 初始化Student训练器
        student_trainer = HugTrainer(
            model=base_model,
            args=training_args,
            train_dataset=pseudo_labeled_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
            data_collator=self.student_data_collator,
        )
        return student_trainer

    def freeze_backbone(self, model: torch.nn.Module, use_pe: bool=False):
        try:
            model.freeze_backbone(use_pe=use_pe)
        except:
            pass
        return model


    def train(self, resume_from_checkpoint=None):
        if not os.path.exists(os.path.join(self.output_dir, "iteration")):
            os.makedirs(os.path.join(self.output_dir, "iteration"))

        teacher_model = self.teacher_base_model
        teacher_model = self.freeze_backbone(teacher_model, use_pe=False)
        teacher_trainer: HugTrainer = self.get_teacher_trainer(base_model=teacher_model, num_train_epochs=self.teacher_training_epoch)

        if resume_from_checkpoint is not None and (os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)) or os.path.isfile(
            os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME))
        ):
            logger.info("*"*80)
            logger.info("* Directly loading the trained teacher model from {} *".format(resume_from_checkpoint))
            logger.info("*"*80)
            print("*"*80)
            logger.info("* Directly loading the trained teacher model from {} *".format(resume_from_checkpoint))
            print("*"*80)
            # 已有teacher模型，直接加载
            teacher_trainer._load_from_checkpoint(resume_from_checkpoint)
        else:

            # 首先对Teacher模型在labeled data上进行full parameter fine-tuning
            logger.info("*"*66)
            logger.info("* Training teacher model over labeled data before self-training. *")
            logger.info("*"*66)
            print("*"*66)
            print("* Training teacher model over labeled data before self-training. *")
            print("*"*66)

            teacher_trainer.train()
            teacher_model.load_state_dict(torch.load(os.path.join(teacher_trainer.state.best_model_checkpoint, "pytorch_model.bin")))
            teacher_trainer.model = teacher_model

        # 原始的训练结果
        metrics = teacher_trainer.evaluate()
        convention_result = metrics["eval_{}".format(self.metric_for_best_model)]

        logger.info("*"*50)
        logger.info("* Conventional fine-tuning metric: {}. *".format(convention_result))
        logger.info("*"*50)
        print("*"*50)
        print("* Conventional fine-tuning metric: {}. *".format(convention_result))
        print("*"*50)

        logger.info("*"*30)
        logger.info("* Starting Self-training ... *")
        logger.info("*"*30)
        print("*"*30)
        print("* Starting Self-training ... *")
        print("*"*30)

        best_test_metric = None
        best_self_training_iteration = None
        best_teacher_model = None

        # 多轮Teacher-Student迭代训练
        for iter in range(self.self_training_epoch):

            logger.info("*"*34)
            logger.info("* Self-training {}-th iteration *".format(iter))
            logger.info("*"*34)
            print("*"*34)
            print("* Self-training {}-th iteration *".format(iter))
            print("*"*34)


            # 获得Teacher模型在测试集上的效果
            if iter > 0:
                teacher_trainer.model = teacher_model
                metrics = teacher_trainer.evaluate()
                # print("metrics=", metrics)

            '''
            e.g., {'eval_loss': 0.6926815509796143, 'eval_accuracy': 0.5234657039711191, 'eval_runtime': 0.7267, 'eval_samples_per_second': 381.161, 'eval_steps_per_second': 48.161, 'epoch': 1.0}
            '''
            logger.info("*"*60)
            logger.info("* The testing result of teacher model is {} result: {} *".format(self.metric_for_best_model, metrics["eval_{}".format(self.metric_for_best_model)]))
            logger.info("*"*60)
            print("*"*60)
            print("* The testing result of teacher model is {} result: {} *".format(self.metric_for_best_model, metrics["eval_{}".format(self.metric_for_best_model)]))
            print("*"*60)

            if best_test_metric is None or best_test_metric < metrics["eval_{}".format(self.metric_for_best_model)]:
                best_test_metric = metrics["eval_{}".format(self.metric_for_best_model)]
                best_self_training_iteration = iter
                best_teacher_model = teacher_model
                logger.info("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
                logger.info("The best teacher model testing result is {}.".format(best_test_metric))
                print("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
                print("The best teacher model testing result is {}.".format(best_test_metric))


            if iter == self.self_training_epoch - 1:
                break


            # # Teacher模型在labeled data上进行parameter-efficient tuning
            # if iter > 0:
            #     logger.info("*"*80)
            #     logger.info("* Tuning the teacher model on labeled data at {}-th self-training iteration. *".format(iter))
            #     logger.info("*"*80)
            #     print("*"*80)
            #     print("* Tuning the teacher model on labeled data at {}-th self-training iteration. *".format(iter))
            #     print("*"*80)

            #     teacher_model = self.freeze_backbone(teacher_model, use_pe=True)
            #     # teacher_trainer: TeacherTrainer = self.get_teacher_trainer(base_model=teacher_model, num_train_epochs=self.teacher_tuning_epoch)
            #     teacher_trainer.train()
            #     teacher_model.load_state_dict(torch.load(os.path.join(teacher_trainer.state.best_model_checkpoint, "pytorch_model.bin")))
            #     teacher_trainer.model = teacher_model

            # Teacher模型在unlabeled data上获取pseudo-labeled data，并根据uncertainty estimation进行采样
            logger.info("*"*72)
            logger.info("Obtaining pseudo-labeled data and uncertainty estimation via MC dropout.")
            logger.info("*"*72)
            print("*"*72)
            print("Obtaining pseudo-labeled data and uncertainty estimation via MC dropout.")
            print("*"*72)

            unlabeled_dataset, y_mean, y_var, y_pred, y_T = teacher_trainer.mc_evaluate(
                unlabeled_dataset=self.unlabeled_dataset,
                unlabeled_data_num=self.unlabeled_data_num,
                T=20,
                num_classes=self.num_classes
                )

            logger.info("*"*42)
            logger.info("* Sampling reliable pseudo-labeled data. *")
            logger.info("*"*42)
            print("*"*42)
            print("* Sampling reliable pseudo-labeled data. *")
            print("*"*42)

            X_batch, y_batch, _ = sample_by_bald_class_easiness(
                tokenizer=self.tokenizer,
                X=unlabeled_dataset,
                y_mean=y_mean,
                y_var=y_var,
                y=y_pred,
                num_samples=int(y_pred.shape[0] * self.pseudo_sample_num_or_ratio) if self.pseudo_sample_num_or_ratio <= 1.0 else int(self.pseudo_sample_num_or_ratio),
                num_classes=self.num_classes,
                y_T=y_T)
            pseudo_labeled_examples = X_batch
            pseudo_labeled_examples["label"] = y_batch

            # 生成pseudo-labeled dataset，并与labeled data混合
            # pseudo_labeled_dataset = DatasetDict()
            pseudo_labeled_dataset = DatasetK.from_dict(pseudo_labeled_examples)
            for i in range(len(self.train_dataset)):
                pseudo_labeled_dataset = pseudo_labeled_dataset.add_item(self.train_dataset[i])

            # 初始化一个新的Student模型，并让Student模型在pseudo-labeled data上进行鲁棒学习
            logger.info("*"*56)
            logger.info("* Training a new student model on pseudo-labeled data. *")
            logger.info("*"*56)
            print("*"*56)
            print("* Training a new student model on pseudo-labeled data. *")
            print("*"*56)

            student_model = self.student_base_model
            student_model = self.freeze_backbone(student_model, use_pe=True)
            student_trainer: HugTrainer = self.get_student_trainer(
                base_model=student_model,
                num_train_epochs=self.student_training_epoch,
                student_learning_rate=self.student_learning_rate,
                pseudo_labeled_dataset=pseudo_labeled_dataset,
                output_dir=os.path.join(self.output_dir, "iteration", "student_iter_{}".format(iter))
            )
            student_trainer.train()
            student_model.load_state_dict(torch.load(os.path.join(student_trainer.state.best_model_checkpoint, "pytorch_model.bin")))

            # 将Student模型参数赋给Teacher，作为下一轮训练的Teacher初始化
            logger.info("*"*64)
            logger.info("* Initializing a new teacher model from trained student model. *")
            logger.info("*"*64)
            print("*"*64)
            print("* Initializing a new teacher model from trained student model. *")
            print("*"*64)
            teacher_model = student_model
            # teacher_trainer = student_trainer
            teacher_trainer: HugTrainer = self.get_teacher_trainer(
                base_model=student_model,
                num_train_epochs=self.teacher_tuning_epoch,
                output_dir=os.path.join(self.output_dir, "iteration", "teacher_iter_{}".format(iter))
            )




        logger.info("********** Finishing Self-training **********")
        logger.info("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
        logger.info("The best teacher model testing result is {}.".format(best_test_metric))
        print("********** Finishing Self-training **********")
        print("The best teacher model at {}-th self-training iteration.".format(best_self_training_iteration))
        print("The best teacher model testing result is {}.".format(best_test_metric))


        # 根据当前最好的Teacher模型，在全部的unlabeled data上打伪标签，并进行mc dropout（样本数量最多不超过50000）
        if self.semi_training_args.post_student_train:

            logger.info("********** Post training **********")
            print("********** Post training **********")

            teacher_trainer: HugTrainer = self.get_teacher_trainer(
                base_model=best_teacher_model,
                num_train_epochs=self.teacher_tuning_epoch,
                output_dir=os.path.join(self.output_dir, "teacher_iter_post")
            )

            unlabeled_dataset, y_mean, y_var, y_pred, y_T = teacher_trainer.mc_evaluate(
                unlabeled_dataset=self.unlabeled_dataset,
                unlabeled_data_num=20480,
                T=5,
                num_classes=self.num_classes
                )

            post_sample_num = int(y_pred.shape[0] * 0.5)

            X_batch, y_batch, _ = sample_by_bald_class_easiness(
                tokenizer=self.tokenizer,
                X=unlabeled_dataset,
                y_mean=y_mean,
                y_var=y_var,
                y=y_pred,
                num_samples=post_sample_num,
                num_classes=self.num_classes,
                y_T=y_T)
            pseudo_labeled_examples = X_batch
            pseudo_labeled_examples["label"] = y_batch
            # 生成pseudo-labeled dataset
            # pseudo_labeled_dataset = DatasetDict()
            pseudo_labeled_dataset = DatasetK.from_dict(pseudo_labeled_examples)


            # 初始化一个新的Student模型，并让Student模型在pseudo-labeled data上进行鲁棒学习
            logger.info("*"*56)
            logger.info("* Training a new student model on pseudo-labeled data. *")
            logger.info("*"*56)
            print("*"*56)
            print("* Training a new student model on pseudo-labeled data. *")
            print("*"*56)

            student_model = self.student_base_model
            student_model = self.freeze_backbone(student_model, use_pe=True)
            student_trainer: HugTrainer = self.get_student_trainer(
                base_model=student_model,
                num_train_epochs=self.student_training_epoch if len(pseudo_labeled_dataset) <= 4096 else int(self.student_training_epoch / 2),
                student_learning_rate=self.student_learning_rate,
                pseudo_labeled_dataset=pseudo_labeled_dataset,
                output_dir=os.path.join(self.output_dir, "student_iter_{}".format(iter))
            )
            student_trainer.train()
            student_model.load_state_dict(torch.load(os.path.join(student_trainer.state.best_model_checkpoint, "pytorch_model.bin")))

            metrics = student_trainer.evaluate()
            post_metric = metrics["eval_{}".format(self.metric_for_best_model)]


        print("*"*68)
        print("Finishing all the processes, the results are shown in the following:")
        print("Conventional fine-tuning {} metric: {}".format(self.metric_for_best_model, convention_result))
        print("Best self-training {} metric: {}".format(self.metric_for_best_model, best_test_metric))
        if self.semi_training_args.post_student_train:
            print("Post training {} metric: {}".format(self.metric_for_best_model, post_metric))
        print("*"*68)

        return TrainOutput(teacher_trainer.state.global_step, 0.0, metrics)
