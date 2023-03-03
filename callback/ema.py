# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 7:33 下午
# @Author  : JianingWang
# @File    : ema.py
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from copy import deepcopy


class ExponentialMovingAveragingCallback(TrainerCallback):
    """
    滑动平均
    相关参数：
    ema: 是否使用ema
    decay: ema的权重
    """

    def __init__(self, decay):
        self.decay = decay
        # 保存影子权重（当前step的每一层的滑动平均权重）
        self.average_model = None
        # 在进行evaluate的时候，保存原始的模型权重，当执行完evaluate后，从影子权重恢复到原始权重
        self.model_weights = None

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        self.average_model = deepcopy(model)
        self.model_weights = deepcopy(model)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        # 更新ema参数
        self.update_parameters(model)
        # 如果需要evaluate则把model更新为ema参数
        if control.should_evaluate:
            self.transfer_weights(model, self.model_weights)
            self.transfer_weights(self.average_model, model)
        # 最后把参数更新为平均值
        if control.should_training_stop:
            self.transfer_weights(self.average_model, model)


    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        # evaluate结束后恢复model参数，如果当前需要保存checkpoint则在save后恢复
        if not control.should_save:
            self.transfer_weights(self.model_weights, model)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        self.transfer_weights(self.model_weights, model)

    @staticmethod
    def transfer_weights(src_model, dst_model):
        for src_param, dst_param in zip(src_model.parameters(), dst_model.parameters()):
            dst_param.detach().copy_(src_param.to(dst_param.device))

    def update_parameters(self, model):
        for p_ema, p_model in zip(self.average_model.parameters(), model.parameters()):
            device = p_ema.device
            p_ema_ = p_ema.detach()
            p_model_ = p_model.detach().to(device)
            src = (1.0 - self.decay) * p_model_ + self.decay * p_ema_
            p_ema_.copy_(src)
