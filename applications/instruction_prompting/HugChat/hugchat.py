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
# from models.language_modeling.causal_lm import GPT2ForCausalLM
import numpy as np
import torch


logging.disable(logging.ERROR)
warnings.filterwarnings("ignore")



class HugChatAPI:
    def __init__(self, model_type, hugchat_model_name_or_path) -> None:
        if model_type not in PRETRAIN_MODEL_CLASSES["causal_lm"].keys():
            raise KeyError(
                "You must choose one of the following model: {}".format(
                    ", ".join(
                        list(PRETRAIN_MODEL_CLASSES["causal_lm"].keys())
                    )
                )
            )
        self.model_type = model_type
        self.model = PRETRAIN_MODEL_CLASSES["causal_lm"][
            self.model_type].from_pretrained(hugchat_model_name_or_path)
        self.tokenizer = TOKENIZER_CLASSES[self.model_type].from_pretrained(hugchat_model_name_or_path)
        self.max_seq_length = 512
        self.max_generation_length = 200
        self.user_name = "Input"
        self.hugchat_name = "Output"
        self.dialog_context = ""
        self.end_string = "\n\n"

    def reset_dialog(self):
        # reset the dialog
        self.dialog_context = ""


    def request(self, text: str):
        text = text.strip()
        if self.dialog_context == "":
            text = "{}: {} ".format(self.user_name, text)
        else:
            text = "{} \n {}: {} ".format(self.dialog_context, self.user_name, text)

        inputs = self.tokenizer(text, return_tensors="pt")

        total_sequences = self.model.generate(
                input_ids=inputs['input_ids'].to(self.model.device),
                attention_mask=inputs['attention_mask'].to(self.model.device),
                max_length=len(inputs['input_ids'][0]) + self.max_generation_length,
                do_sample=True,
                num_beams=3,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response_text: str = self.tokenizer.decode(total_sequences[0][len(inputs["input_ids"][0]):])
        response_text = response_text.strip()

        if response_text.startswith(self.hugchat_name + ":"):
            response_text = response_text[len(self.hugchat_name + ":"):].strip()

        # clip the remaining tokens
        try:
            index = response_text.index(self.end_string)
        except ValueError:
            response_text += self.end_string
            index = response_text.index(self.end_string)

        response_text = response_text[:index + 1]

        self.dialog_context = "{} \n {}: {} ".format(text, self.hugchat_name, response_text)
        self.dialog_context = self.dialog_context[-self.max_seq_length:] # remove over length

        return response_text

def print_hello():
    length = 82
    print("+" + "-"*(length - 2) + "+")
    print("|" + " "*(length - 2) + "|")
    print(" " + " "*int((length - 2 - 25)/2) + "🤗 Welcome to use HugNLP!" + " "*int((length - 2 - 25)/2)  + " ")
    print("" + " "*(length) + "")
    print(" " + " "*int((length - 2 - 31)/2) + "You can chat with HugChat Now!" + " "*int((length - 2 - 31)/2)  + " ")
    print("" + " "*(length) + "")
    print("" + " "*int((length - 2 - 32)/2) + "https://github.com/wjn1996/HugNLP" + " "*int((length - 2 - 33)/2)  + "")
    print("|" + " "*(length - 2) + "|")
    print("+" + "-"*(length - 2) + "+")
    print(end="")


def main():
    model_type = "gpt2"
    hugchat_model_name_or_path = "wjn1996/hugnlp-hugchat-gpt2"
    hugchat = HugChatAPI(model_type, hugchat_model_name_or_path)

    # Chats

    print_hello()

    print("Guideline: Exit: no print. New chat: input '<new chat>' \n\n")

    end_string = "\n\n"

    while True:
        input_text = input("Human >>> ")
        if not input_text:
            print("exit...")
            break

        if input_text == "<new chat>":
            hugchat.reset_dialog()
            print("You can make a new chat topic now! \n")
            continue

        response_text = hugchat.request(input_text)
        print("HugChat: " + response_text, end="\n")


if __name__ == "__main__":
    main()
