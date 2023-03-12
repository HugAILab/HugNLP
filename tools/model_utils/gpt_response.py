import os
import sys
import torch
import openai
import time

"""
调用或转换输出结果为OpenAI GPT模式
"""
class GPTResponse:

    def __init__(self, model_type: str, data_path: str) -> None:
        assert model_type in ["gpt2", "gpt3"]
        self.model_type = model_type
        if self.model_type == "gpt3":

            with open(os.path.join(data_path, 'openai_key.txt'), 'r') as f:
                key = f.readline().strip()
                openai.api_key = key

    def call_for_gpt3_response(self, prompt, l, model_name, temp=0, num_log_probs=None, echo=False, n=None):
        """
        call GPT-3 API until result is provided and then return it
        """
        response = None
        received = False
        while not received:
            try:
                response = openai.Completion.create(engine=model_name, prompt=prompt, max_tokens=l, temperature=temp,
                                                    logprobs=num_log_probs, echo=echo, stop='\n', n=n)
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False

                print("API error:", error)
                time.sleep(1)
        return response

    def call_for_gpt2_response(self, gpt2_tokenizer, logits, total_sequences, num_log_probs=None, echo=False, n=None):
        """
        Obtain the prediction logits from gpt2 in local, and convert it to the value that can match the response from OpenAI
        """
        if not echo:
            # get the top tokens and probs for the generated l tokens
            probs = torch.softmax(logits[:,-l-1:], dim=2).cpu()
        else:
            # get the top tokens and probs for the context and the generated l tokens
            probs = torch.softmax(logits, dim=2).cpu()
        top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
        logprobs = torch.log(probs)
        top_log_probs = torch.log(top_probs)

        # create the return value to resemble OpenAI
        return_json = {}
        choices = []
        # print("="*50)
        for batch_id in range(len(logits)):
            curr_json = {}
            # text is just the optional context and next l tokens
            if not echo:
                curr_json['text'] = gpt2_tokenizer.decode(total_sequences[batch_id][-l:], skip_special_tokens=True)
            else:
                curr_json['text'] = gpt2_tokenizer.decode(total_sequences[batch_id], skip_special_tokens=True)

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
                        curr_json['logprobs']['tokens'].append(gpt2_tokenizer.decode([current_element_top_tokens[0]]))
                        # token_logprobs is a list of the logprob of the top token at each position
                        curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                        # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                        temp = {}
                        for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                            temp[gpt2_tokenizer.decode(token.item())] = log_prob.item()
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
                            temp[gpt2_tokenizer.decode(token.item())] = log_prob.item()
                        curr_json['logprobs']['top_logprobs'].append(temp)
                    for index in range(len(probs[batch_id])):
                        curr_json['logprobs']['tokens'].append(gpt2_tokenizer.decode([total_sequences[batch_id][index]]))
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
