#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 HugNLP. All rights reserved.
# The founder is Jianing Wang, a Ph.D student in ECNU.

"""A simple shell chatbot implemented with lmflow APIs.
"""
import logging
import warnings
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
from models import PRETRAIN_MODEL_CLASSES
from models import TOKENIZER_CLASSES
# from transformers import AutoTokenizer
import numpy as np
import torch


logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")

CANDIDATE_CAUSAL_MODELS = ["gpt2", "opt", "llama", "glm"]

class HugChatAPI:
    def __init__(self, model_type, hugchat_model_name_or_path) -> None:
        if model_type not in CANDIDATE_CAUSAL_MODELS:
            raise KeyError(
                "You must choose one of the following model: {}".format(
                    ", ".join(
                        list(CANDIDATE_CAUSAL_MODELS)
                    )
                )
            )
        print("Loading chat model from {}".format(hugchat_model_name_or_path))
        self.model_type = model_type
        # self.model = PRETRAIN_MODEL_CLASSES["auto_causal_lm"][self.model_type].from_pretrained(hugchat_model_name_or_path)
        self.model = PRETRAIN_MODEL_CLASSES["auto_causal_lm"].from_pretrained(hugchat_model_name_or_path)
        if torch.cuda.is_available():
            self.model.cuda(0)
        self.tokenizer = TOKENIZER_CLASSES["auto"].from_pretrained(hugchat_model_name_or_path, use_fast=False)
        # self.tokenizer = AutoTokenizer.from_pretrained(hugchat_model_name_or_path)
        self.max_seq_length = 512
        self.max_generation_length = 200
        self.user_name = "[Human]"
        self.hugchat_name = "[HugChat]"
        self.prompt_before_use = "Your name is HugChat. You are developed by HugNLP library and belong to HugAI Lab and ECNU. Your founder is Jianing Wang. The following is the chat between human and HugChat. \n\n "
        self.dialog_context = self.prompt_before_use
        self.end_string = ["\n\n", "\n\n [HugChat]", "\n\n [Human]", "[Human]:", "<|endoftext|>", "<|endoftext|></s>"]
        self.stop_token = {
            "gpt2": "\n\n",
            "opt": "\n\n",
            "llama": None,
            "glm": None,
        }

    def reset_dialog(self):
        # reset the dialog
        self.dialog_context = self.prompt_before_use


    def request(self, text: str):
        text = text.strip()
        # text = "{} \n {}: {} ".format(self.dialog_context, self.user_name, text)
        text = self.dialog_context + self.user_name + ": " + text
        inputs = self.tokenizer(text, return_tensors="pt")

        total_sequences = self.model.generate(
                input_ids=inputs['input_ids'].to(self.model.device),
                attention_mask=inputs['attention_mask'].to(self.model.device),
                # max_length=len(inputs['input_ids'][0]) + self.max_generation_length,
                max_length=1024,
                do_sample=True,
                temperature=0.7,
                # num_beams=3,
                top_k=40, 
                top_p=0.8, 
                repetition_penalty=1.02,
                eos_token_id=self.tokenizer.encode(self.stop_token[self.model_type])[0],
                pad_token_id=self.tokenizer.pad_token_id
            )

        response_text: str = self.tokenizer.decode(total_sequences[0][len(inputs["input_ids"][0]):])
        response_text = response_text.strip()
        # response_text = response_text.strip().replace("<|endoftext|></s>", "")
        # if response_text[-1] not in [".", "?", "!"]:
        #     response_text += "."

        if response_text.startswith(self.hugchat_name + ":"):
            response_text = response_text[len(self.hugchat_name + ":"):].strip()

        # print("response_text=", response_text)
        # clip the remaining tokens
        # try:
        #     # index = response_text.index(self.end_string[0])
        #     index = len(response_text) - 1
        #     for end_string in self.end_string:
        #         index = min(response_text.index(end_string), index)
        # except ValueError:
        #     response_text += self.end_string[0]
        #     index = response_text.index(self.end_string[0])
        index = len(response_text) - 1
        for end_string in self.end_string:
            try:
                cur_index = response_text.index(end_string)
                # print("index=", index)
            except ValueError:
                response_text += end_string
                cur_index = response_text.index(end_string)
                # print("index=", index)
            index = min(cur_index, index)
        
        response_text = response_text[:index]
        # print("response_text=", response_text)

        self.dialog_context = "{} \n {}: {} ".format(text, self.hugchat_name, response_text.strip())
        self.dialog_context = self.dialog_context[-self.max_seq_length:] # remove over length

        return response_text.strip()

def print_hello():
    length = 82
    print("+" + "-"*(length - 2) + "+")
    print("|" + " "*(length - 2) + "|")
    print(" " + " "*int((length - 2 - 25)/2) + "🤗 Welcome to use HugNLP!" + " "*int((length - 2 - 25)/2)  + " ")
    print("" + " "*(length) + "")
    print(" " + " "*int((length - 2 - 31)/2) + "You can chat with HugChat Now!" + " "*int((length - 2 - 31)/2)  + " ")
    print("" + " "*(length) + "")
    print("" + " "*int((length - 2 - 32)/2) + "https://github.com/HugAILab/HugNLP" + " "*int((length - 2 - 33)/2)  + "")
    print("|" + " "*(length - 2) + "|")
    print("+" + "-"*(length - 2) + "+")
    print(end="")


def main():
    # model_type = "gpt2"
    # hugchat_model_name_or_path = "wjn1996/hugnlp-hugchat-gpt2"
    # hugchat_model_name_or_path = "./outputs/instruction/causal_lm_gpt2/gpt2-large/gpt2-large"
    # hugchat_model_name_or_path = "./outputs/instruction/causal_lm_gpt2/gpt2-xl/gpt2-xl"
    # hugchat_model_name_or_path = "./outputs/instruction/causal_lm_gpt2/gpt2/gpt2"

    model_type = "opt"
    hugchat_model_name_or_path = "./outputs/instruction/causal_lm_opt/opt-1.3b/opt-1.3b"
    # print("Please select a ChatGPT-like model ")

    hugchat = HugChatAPI(model_type, hugchat_model_name_or_path)

    # Chats

    print_hello()

    print("Guidelines: \n- Have a new chat, please input '<clear>'. \n- Use a new chat model, please ipnut '<model_name_or_path=xxx>'. \n- Stop the chat and exit, please input carriage return. \n\n")

    while True:
        input_text = input("Human >>> ")
        if not input_text:
            print("exit...")
            break

        if input_text == "<clear>":
            hugchat.reset_dialog()
            print("You can make a new chat topic now! \n")
            continue

        if "<model_name_or_path=" in input_text:
            hugchat.reset_dialog()
            hugchat_model_name_or_path = input_text.replace("<model_name_or_path=", "")
            hugchat_model_name_or_path = hugchat_model_name_or_path[:-1]
            hugchat = HugChatAPI(model_type, hugchat_model_name_or_path)
            continue


        response_text = hugchat.request(input_text)
        print("HugChat: " + response_text, end="")
        print("\n")


if __name__ == "__main__":
    main()
