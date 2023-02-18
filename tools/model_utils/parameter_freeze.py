# -*- coding: utf-8 -*-
# @Time    : 2023/02/18 14:07 上午
# @Author  : JianingWang
# @File    : parameter_freeze.py

import torch


"""
This is use for parameter fixing and unfreezing, which can be viewed as parameter-efficient settings.
"""
class ParameterFreeze():
    # freeze all parameters
    def freeze_lm(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            param.requires_grad = False
        return model

    # freeze all parameters without cls / mlm head
    def freeze_lm_encoder(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if 'lm_head' in name or ('cls' in name):
                print(name)
                continue
            param.requires_grad = False
        return model

    # freeze all parameters without bias
    def freeze_lm_finetune_bias(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if "bias" in name:
                print(name)
                continue
            param.requires_grad = False
        return model

    # freeze the component that user defined
    def freeze_lm_component(self, model: torch.nn.Module, component: str):
        if 'attention' in component:
            for name, param in model.named_parameters():
                if 'attention' in name:
                    if 'output' in component:
                        if 'output' in name:
                            continue
                    else:
                        continue
                param.requires_grad = False
            model = self.unfreeze_classification_head(model)
        elif 'feedforward' in component:
            for name, param in model.named_parameters():
                if 'dense' in name and 'attention' not in name:
                    if 'output' in component:
                        if 'output' in name:
                            continue
                    else:
                        if 'intermediate' in component:
                            if 'intermediate' in name:
                                continue
                param.requires_grad = False
            model = self.unfreeze_classification_head(model)
        elif component == 'adapter':
            for name, param in model.named_parameters():
                if 'adapter' in name:
                    continue

                param.requires_grad = False
            model = self.unfreeze_classification_head(model)
        elif 'embedding' in component:
            for name, param in model.named_parameters():
                if 'embedding' in name:
                    continue

                param.requires_grad = False
            model = self.unfreeze_classification_head(model)
        elif 'bias' in component:
            for name, param in model.named_parameters():
                if 'bias' in name:
                    continue
                param.requires_grad = False
            model = self.unfreeze_classification_head(model)
        elif 'head' in component:
            for name, param in model.named_parameters():
                param.requires_grad = False
            model = self.unfreeze_classification_head(model)

        elif "prompt_emb" in component:
            for name, param in model.named_parameters():
                if 'prompt_emb' in name:
                    continue
                param.requires_grad = False
        return model

    # unfreeze cls head
    def unfreeze_classification_head(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if 'lm_head' in name or ('cls' in name) or ('classifier' in name):
                param.requires_grad = True
        return model

    # freeze k layers
    def freeze_lm_k_layers(self, model: torch.nn.Module, k):
        keep_layers = []
        update_parameters = []
        for i in range(k):
            keep_layers.append('layer.'+str(23-i))

        for name, param in model.named_parameters():
            update = False
            for layer_num in keep_layers:
                if layer_num in name:
                    if 'dense' in name and 'attention' not in name:
                        if 'output' in name:
                            print(name)
                            update_parameters.append(name)
                            update = True

            if not update:
                param.requires_grad = False
        model = self.unfreeze_classification_head(model)
        return model


    def unfreeze_lm(self, model: torch.nn.Module):
        for param in model.parameters():
            param.requires_grad = True
        return model
