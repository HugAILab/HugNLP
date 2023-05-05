# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 8:02 p.m.
# @Author  : Jianing Wang
# @File    : calibrate.py

import os
import numpy as np
import torch

"""
Use LM to classify label words for calibrating CLS
"""
class CLSCalibrator:
    pass

"""
Use Causal LM to generate label words for calibrating CLS
e.g., use gpt2 to generate a label word with in-context prompts, and calibrate for the prediction.
Paper: http://proceedings.mlr.press/v139/zhao21c.html
"""
class CausalCLSCalibrator:

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def calibrate(self, all_label_probs, content_free_examples, label2id, mode="diagonal_W"):
        """Perform calibration for de-biasing and obtain calibrated probability"""
        p_cf = self.get_content_free_prediction(content_free_examples, label2id)

        num_classes = all_label_probs.shape[1]
        if p_cf is None:
            # do not calibrate
            W = np.identity(num_classes)
            b = np.zeros([num_classes, 1])
        else:
            # calibrate
            if mode == "diagonal_W":
                W = np.linalg.inv(np.identity(num_classes) * p_cf)
                b = np.zeros([num_classes, 1])
            elif mode == "identity_W":
                W = np.identity(num_classes)
                b = -1 * np.expand_dims(p_cf, axis=-1)
            else:
                assert False


        all_calibrate_label_probs = list()
        for label_probs in all_label_probs:
            label_probs = label_probs / np.sum(label_probs) # normalize to 1
            calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
            all_calibrate_label_probs.append(calibrate_label_probs.squeeze().tolist())
        return np.array(all_calibrate_label_probs)


    def get_content_free_prediction(self, content_free_examples, label2id: dict):
        """Query model with content free input, return its prediction probability for each label"""

        all_p_y = []
        for content_free_example in content_free_examples:

            content_free_prompt = content_free_example["content_free_prompt"]
            p_y = [0] * len(label2id)
            for answers, i in label2id.items():
                prob = 0
                for a in answers:
                    prob += np.exp(self.get_causal_cls_prediction(content_free_prompt + " " + a, 0, echo=True, num_log_probs=1)['choices'][0]['logprobs']['token_logprobs'][-1])
                p_y[i] = prob
            all_p_y.append(p_y)

        p_y = np.mean(np.array(all_p_y), axis=0)
        p_y = p_y / np.sum(p_y) # normalize
        return p_y


    def get_causal_cls_prediction(self, prompt, l=10, num_log_probs=None, echo=False):
        ''' This function runs GPT-2 locally but places the outputs into an json that looks just like the one
        provided by the OpenAI API. '''
        if isinstance(prompt, str):
            prompt = [prompt] # the code below assumes a list
        input_ids = self.tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)

        if l + len(input_ids['input_ids'][0]) > 1020:
            m = l + len(input_ids['input_ids'][0]) - 1024
            input_ids['input_ids'] = torch.Tensor([input_ids['input_ids'][0][m:].numpy()]).long()
            input_ids['attention_mask'] = torch.Tensor([input_ids['attention_mask'][0][m:].numpy()]).long()

        # greedily generate l tokens
        # print("l=", l)
        if l > 0:
            # the generate function can handle left padded inputs automatically in HF
            # total_sequences is now the input + possible generated output
            # print("l + len(input_ids[input_ids][0]=", l + len(input_ids['input_ids'][0]))
            total_sequences = self.model.generate(
                input_ids=input_ids['input_ids'].to(self.model.device),
                attention_mask=input_ids['attention_mask'].to(self.model.device),
                max_length=l + len(input_ids['input_ids'][0]),
                do_sample=False
                )
        else:
            assert echo == True and l == 0
            total_sequences = input_ids['input_ids'].to(self.model.device)
        # print("="*50)
        # print("total_sequences=", total_sequences) [batch, len+l]
        # print("total_sequences.shape=", total_sequences.shape)

        # they want the probs of the top tokens
        if num_log_probs is not None:
            # we are left padding, so we need to adjust the position IDs
            attention_mask = (total_sequences != 50256).float()
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # get the logits for the context and the next l tokens
            logits = self.model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu()
            if not echo:
                # get the top tokens and probs for the generated l tokens
                probs = torch.softmax(logits[:,-l-1:], dim=2).cpu()
            else:
                # get the top tokens and probs for the context and the generated l tokens
                probs = torch.softmax(logits, dim=2).cpu()
            top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
            logprobs = torch.log(probs)
            top_log_probs = torch.log(top_probs)
            # print("top_log_probs=", top_log_probs)
            # print("top_log_probs.shape=", top_log_probs.shape) # [1, 2, 100] [batch, 2, api_num_log_prob]

        # create the return value to resemble OpenAI
        return_json = {}
        choices = []
        # print("="*50)
        for batch_id in range(len(prompt)):
            curr_json = {}
            # text is just the optional context and next l tokens
            if not echo:
                curr_json['text'] = self.tokenizer.decode(total_sequences[batch_id][-l:], skip_special_tokens=True)
            else:
                curr_json['text'] = self.tokenizer.decode(total_sequences[batch_id], skip_special_tokens=True)

            # fill the return json with the top tokens and probs to match the OpenAI return value.
            if num_log_probs is not None:
                curr_json['logprobs'] = {}
                curr_json['logprobs']['top_logprobs'] = []
                curr_json['logprobs']['token_logprobs'] = []
                curr_json['logprobs']['tokens'] = []
                if not echo:
                    # cutoff the -1 here because the probs are shifted one over for LMs
                    for current_element_top_log_probs, current_element_top_tokens in zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1]):
                        # tokens is a list of the top token at each position
                        curr_json['logprobs']['tokens'].append(self.tokenizer.decode([current_element_top_tokens[0]]))
                        # token_logprobs is a list of the logprob of the top token at each position
                        curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                        # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                        temp = {}
                        for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                            temp[self.tokenizer.decode(token.item())] = log_prob.item()
                        curr_json['logprobs']['top_logprobs'].append(temp)
                else:
                    # same as not above but small tweaks
                    # we add null to the front because for the GPT models, they have null probability for the first token
                    # (for some reason they don't have an beginning of sentence token)
                    curr_json['logprobs']['top_logprobs'].append('null')
                    # cutoff the -1 here because the probs are shifted one over for LMs
                    for index, (current_element_top_log_probs, current_element_top_tokens) in enumerate(zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1])):
                        # skip padding tokens
                        if total_sequences[batch_id][index].item() == 50256:
                            continue
                        temp = {}
                        for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                            temp[self.tokenizer.decode(token.item())] = log_prob.item()
                        curr_json['logprobs']['top_logprobs'].append(temp)
                    for index in range(len(probs[batch_id])):
                        curr_json['logprobs']['tokens'].append(self.tokenizer.decode([total_sequences[batch_id][index]]))
                    curr_json['logprobs']['token_logprobs'].append('null')
                    for index, log_probs_token_position_j in enumerate(logprobs[batch_id][:-1]):
                        # probs are left shifted for LMs
                        curr_json['logprobs']['token_logprobs'].append(log_probs_token_position_j[total_sequences[batch_id][index+1]])

            choices.append(curr_json)
            # print("curr_json=", curr_json)
            '''
            e.g.,
            num_tokens_to_predict=1
            curr_json= {
                'text': ' I', # 当前生成的top词
                'logprobs': {'top_logprobs': [{' I': -3.4267239570617676, '\n': -3.5073862075805664, ...], # top100词及其socre
                'token_logprobs': [-3.4267239570617676], # 当前top词的score
                'tokens': [' I']}
            }
            num_tokens_to_predict=2
            curr_json= {
                'text': '\nThe', # 如果指定生成两个词，则为两个词
                'logprobs': {'top_logprobs': [ # 两个位置对应的预测的score
                    {'\n': -3.186706304550171, '\xa0': -3.222092390060425, ' We': -6.781067848205566, ...},
                    {'The': -2.5251243114471436, '"': -2.857935667037964, ...],
                'token_logprobs': [-3.186706304550171, -2.5251243114471436], # 生成的词的score
                'tokens': ['\n', 'The']}
            }
            '''
        return_json['choices'] = choices
        # print("="*50)
        # print("return_json=", return_json)
        return return_json
