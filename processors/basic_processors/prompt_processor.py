from typing import List, Optional
from transformers import AutoTokenizer
from config import DataTrainingArguments
from tools.processing_utils.tokenizer.tokenizer_utils import get_special_token_mapping
"""
For prompt-based classification, add task-specific discrete template for each example.
"""

"""
The base prompt processor
"""
class PromptBaseProcessor:
    def __init__(
        self,
        data_args: DataTrainingArguments,
        task_name: str,
        sentence1_key: str,
        sentence2_key: str,
        template: List[Optional[dict]],
        label_words_mapping: dict,
        instruction: Optional[dict] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:

        self.data_args = data_args  # DataTrainingArguments
        self.task_name = task_name
        self.sentence1_key = sentence1_key
        self.sentence2_key = sentence2_key
        self.template = template
        self.instruction = instruction
        self.label_words_mapping = label_words_mapping
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            self.special_token_mapping = get_special_token_mapping(
                tokenizer=self.tokenizer)
        """
        template: [<template for sentence1>, <template for sentence2>]
        <template for sentence1>: {"prefix_template": xx, "suffix_template": xx}
        <template for sentence2>: {"prefix_template": xx, "suffix_template": xx}

        e.g.,
        task_name: mnli
        sentence1_key: premise
        sentence2_key: hypothesis
        template: [None, {"prefix_template": " ? <mask> , ", "suffix_template": ""}],

        """

    def set_tokenizer(self, tokenizer: AutoTokenizer):
        assert tokenizer is not None
        self.tokenizer = tokenizer
        self.special_token_mapping = get_special_token_mapping(
            tokenizer=self.tokenizer)

    def add_prompt_into_example(self, examples):
        def replace_mask_token(template):
            return template.replace("<mask>", self.tokenizer.convert_ids_to_tokens(self.special_token_mapping["mask"]))

        sequence1_prefix_template = replace_mask_token(
            self.template[0]["prefix_template"] if self.template[0] is not None else ""
        )
        sequence1_suffix_template = replace_mask_token(
            self.template[0]["suffix_template"] if self.template[0] is not None else ""
        )
        sequence2_prefix_template = replace_mask_token(
            self.template[1]["prefix_template"] if self.template[1] is not None else ""
        )
        sequence2_suffix_template = replace_mask_token(
            self.template[1]["suffix_template"] if self.template[1] is not None else ""
        )

        example_num = len(examples[self.sentence1_key])
        for example_id in range(example_num):
            sequence1 = examples[self.sentence1_key][example_id]
            if self.sentence2_key is None:
                sequence1 = sequence1[:self.data_args.max_seq_length - len(sequence1_suffix_template) - 10]
            else:
                sequence1 = sequence1[:self.data_args.max_seq_length // 2 - len(sequence1_suffix_template) - 10]
            examples[self.sentence1_key][example_id] = "{}{}{}".format(sequence1_prefix_template, sequence1, sequence1_suffix_template)

            if self.sentence2_key is not None:
                sequence2 = examples[self.sentence2_key][example_id]
                sequence2 = sequence2[:self.data_args.max_seq_length - len(sequence1) - len(sequence1_prefix_template) - len(sequence1_suffix_template) - len(sequence2_prefix_template) - 10]
                examples[self.sentence2_key][example_id] = "{}{}{}".format(sequence2_prefix_template, sequence2, sequence2_suffix_template)
        return examples


    def obtain_label_word_list(self):
        label_to_word = self.label_words_mapping  # e.g., {"0": ["great"], "1": [bad]}
        label_list = list(label_to_word.keys())
        label_to_word = {
            label: label_word[0] if type(label_word) == list else label_word
            for label, label_word in self.label_words_mapping.items()
        }

        for key in label_to_word:
            # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
            if label_to_word[key][0] not in ["<", "[", ".", ","]:
                # Make sure space+word is in the vocabulary
                # assert len(self.tokenizer.tokenize(" " + label_to_word[key])) == 1
                label_to_word[key] = self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(" " + label_to_word[key])[0])
            else:
                label_to_word[key] = self.tokenizer.convert_tokens_to_ids(
                    label_to_word[key])

        if len(label_list) > 1:
            label_word_list = [label_to_word[label] for label in label_list]
        else:
            # Regression task
            # "0" represents low polarity and "1" represents high polarity.
            label_word_list = [label_to_word[label] for label in ["0", "1"]]
        return label_word_list



"""
The prompt processor for instruction / in-context tuning
"""
class InstructionPromptProcessor(PromptBaseProcessor):

    def __init__(
        self, data_args: DataTrainingArguments,
        task_name: str,
        sentence1_key: str,
        sentence2_key: str,
        template: List[Optional[dict]],
        label_words_mapping: dict,
        instruction: Optional[dict] = None,
        tokenizer: Optional[AutoTokenizer] = None
        ) -> None:
        super().__init__(data_args, task_name, sentence1_key, sentence2_key, template, label_words_mapping, instruction, tokenizer)
        assert self.instruction is not None, "If you choose instruction prompt, you must define a instruction file."
        self.instruction_prompt = self.instruction["instruction"] if "instruction" in self.instruction.keys() else ""
        self.input_prompt = self.instruction["input_prompt"] if "input_prompt" in self.instruction.keys() else ""
        self.output_prompt = self.instruction["output_prompt"] if "output_prompt" in self.instruction.keys() else ""
        self.input_prompt = self.input_prompt.strip()
        self.output_prompt = self.output_prompt.strip()

    def construct_incontext_prompt(
        self,
        sentence1_key: str,
        sentence2_key: str,
        incontext_examples: List[dict],
        eval_example: dict
    ):
        """
        generate prompt with multiple in-context examples.
        prompt_prefix = "What are follows emotions?"
        q_prefix = "Input: "
        a_prefix = "Output: "
        prompt = "What are follows emotions? Input: The book is very nice.\n Output: great.\n\n Input: I never eat chocolate!\n Output: bad.\n\n Input: This film is wonderful.\n Output: "

        incontext_examples[0], eval_example:
        {
            "sentence1": "xxx",
            "label": "xx", # may be a label id
            "target": "xx" # the label name or answer text
        }

        """
        prompt = self.instruction_prompt + "\n\n"
        for incontext_example in incontext_examples:
            s = incontext_example[sentence1_key] + (" " + incontext_example[sentence2_key] if sentence2_key in eval_example.keys() and incontext_example[sentence2_key] is not None else "")
            l = incontext_example["target"]
            label_word = self.label_words_mapping[l][0] if isinstance(self.label_words_mapping[l], list) else self.label_words_mapping[l]
            prompt += self.input_prompt + " "
            prompt += s + "\n"
            prompt += self.output_prompt + " "
            prompt += label_word + "\n\n"

        eval_s = eval_example[sentence1_key] + (" " + eval_example[sentence2_key] if sentence2_key in eval_example.keys() and eval_example[sentence2_key] is not None else "")
        prompt += self.input_prompt + " "
        prompt += eval_s + "\n"

        prompt += self.output_prompt # GPT models do not want a trailing space, so we cut off -1
        return prompt
