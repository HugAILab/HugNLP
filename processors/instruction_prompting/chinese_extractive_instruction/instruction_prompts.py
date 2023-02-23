import random
from copy import deepcopy
from typing import List, Dict
NO_ANSWER = ''
SEP = '‖'


class Instruction(object):
    def __init__(self, data_name: str, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str, max_length: int = 510):
        self.data_name = data_name
        self.data_list = data_list
        self.verbalizer = verbalizer
        self.instruction = instruction
        self.keys_order = keys_order
        self.data_type = data_type
        self.max_length = max_length

    def transform2instruction(self):
        raise NotImplementedError

    def get_start(self, example):
        """获取序列标注答案的起始位置"""
        if example['target'] in ['', NO_ANSWER]:
            return 0
        start = 0
        verbalizer = example['verbalizer'].split(SEP)
        target = example['target']
        for n in range(0, verbalizer.index(target)):
            start = start + len(verbalizer[n]) + 1
        start += example['instruction'].index(example['verbalizer'])
        return start


class NERInstruction(Instruction):
    def __init__(self, data_name: str, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(NERInstruction, self).__init__(data_name, data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["target"] = example["entities"]
            example["entity_type"] = self.verbalizer[example["entity_type"]]
            example["verbalizer"] = SEP.join(list(set(self.verbalizer.values())))
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            example['start'] = self.get_start(example)
            examples.append(example)
        return examples

    def get_start(self, example):
        out = []
        content_index = example['instruction'].index(example['context'])
        for target in example['target']:
            index = []
            for i in range(0, len(example['context'])):
                if example['context'][i:i + len(target)] == target:
                    index.append(content_index + i)
            out.append(index)
        return out


class MRCInstruction(Instruction):
    def __init__(self, data_name: str, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(MRCInstruction, self).__init__(data_name, data_list, verbalizer, instruction, keys_order, data_type)
        self.NO_ANSWER = NO_ANSWER

        if self.data_name == 'drcd':
            import opencc
            self.converter = opencc.OpenCC('t2s.json')

    def process_answer(self, answer_text):
        if len(answer_text) == 0:
            return self.NO_ANSWER
        answer_text = answer_text[0] if type(answer_text) is list else answer_text
        if answer_text == "":
            return self.NO_ANSWER
        return answer_text

    def transform2instruction(self):
        # TODO 处理序列过长的分割问题
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            try:
                # drcd 繁体转简体
                if self.data_name == 'drcd':
                    example['context'] = self.converter.convert(example['context'])
                    example['question'] = self.converter.convert(example['question'])
                    example['answer'] = [self.converter.convert(a) for a in example['answer']]

                example["target"] = self.process_answer(example["answer"])
            except Exception as e:
                print(e)
                print(sample)
                input()
            slots = [example[k] for k in self.keys_order]
            example["instruction"] = self.instruction.format(*slots)
            example["data_type"] = self.data_type
            example['start'] = self.get_start(example)
            if example['start'] != 0:
                assert example['instruction'][example['start']:example['start'] + len(example['target'])] == example['target']
            examples.append(example)
        return examples

    def get_start(self, example):
        if example["target"] in ['', NO_ANSWER]:
            return 0
        content_index = example['instruction'].index(example['context'])
        target_index = example['context'].index(example['target'])
        return content_index+target_index


# TODO 1. 将multi choice改为分类问题评价 2. 处理长度过长问题
class MultiChoiceInstruction(Instruction):
    def __init__(self, data_name: str, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(MultiChoiceInstruction, self).__init__(data_name, data_list, verbalizer, instruction, keys_order, data_type)
        self.NO_ANSWER = NO_ANSWER

    def process_answer(self, answer_text):
        if answer_text == "":
            return self.NO_ANSWER
        return answer_text

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            if self.data_name == 'c3':
                example["target"] = self.process_answer(example["answer"][0])
                example["choice"] = SEP.join(example["choice"])
            elif self.data_name == 'dureader_yesno':
                example["target"] = self.process_answer(self.verbalizer[example["label"].lower()])
                example["choice"] = SEP.join(list(set(self.verbalizer.values())))
            elif self.data_name == 'cail_yesno':
                example["target"] = self.process_answer(self.verbalizer[example["answer"][0].lower()])
                example["choice"] = SEP.join(list(set(self.verbalizer.values())))
            else:
                example["target"] = self.process_answer(example["answer"])
                example["choice"] = SEP.join(example["choice"])
            
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            example['start'] = self.get_start(example)
            assert example['instruction'][example['start']:example['start']+len(example['target'])] == example['target']
            # if a != example['target']:
            #     print(a, example['target'])
            examples.append(example)
        return examples

    def get_start(self, example):
        if example['target'] in ['', NO_ANSWER]:
            return 0
        start = 0
        verbalizer = example['choice'].split(SEP)
        target = example['target']
        for n in range(0, verbalizer.index(target)):
            start = start + len(verbalizer[n]) + 1
        start += example['instruction'].index(example['choice'])
        return start

class YesNoInstruction(Instruction):
    def __init__(self, data_name: str, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(YesNoInstruction, self).__init__(data_name, data_list, verbalizer, instruction, keys_order, data_type)
        self.NO_ANSWER = NO_ANSWER

    def process_answer(self, answer_text):
        if answer_text == "":
            return self.NO_ANSWER
        return answer_text

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["target"] = self.process_answer(self.verbalizer[example["answer"][0].lower()])
            example["verbalizer"] = SEP.join(list(set(self.verbalizer.values())))
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            example['start'] = self.get_start(example)
            assert example['instruction'][example['start']:example['start']+len(example['target'])] == example['target']
            # if a != example['target']:
            #     print(a, example['target'])
            examples.append(example)
        return examples

    def get_start(self, example):
        if example['target'] in ['', NO_ANSWER]:
            return 0
        start = 0
        verbalizer = example['verbalizer'].split(SEP)
        target = example['target']
        for n in range(0, verbalizer.index(target)):
            start = start + len(verbalizer[n]) + 1
        start += example['instruction'].index(example['verbalizer'])
        return start

class SUMMInstruction(Instruction):
    def __init__(self, data_name: str, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(SUMMInstruction, self).__init__(data_name, data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["target"] = example["summary"]
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            examples.append(example)
        return examples


class KEYSInstruction(Instruction):
    def __init__(self, data_name, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(KEYSInstruction, self).__init__(data_name, data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["target"] = "，".join(example["keys"])
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            examples.append(example)
        return examples


class NLIInstruction(Instruction):
    def __init__(self, data_name, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(NLIInstruction, self).__init__(data_name, data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["target"] = self.verbalizer.get(example["label"], '')
            example["verbalizer"] = SEP.join(list(set(self.verbalizer.values())))
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            example['start'] = self.get_start(example)
            assert example['instruction'][example['start']:example['start'] + len(example['target'])] == example['target']
            examples.append(example)
        return examples


class STSInstruction(Instruction):
    def __init__(self, data_name, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(STSInstruction, self).__init__(data_name, data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            label = example["label"]
            example["target"] = self.verbalizer.get(str(label), '')
            example["verbalizer"] = SEP.join(list(set(self.verbalizer.values())))
            example['text_a'] = example['text_a'][:int((self.max_length - 20) / 2)]
            example['text_b'] = example['text_b'][:int((self.max_length - 20) / 2)]
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            example['start'] = self.get_start(example)
            assert example['instruction'][example['start']:example['start'] + len(example['target'])] == example['target']
            examples.append(example)
        return examples


class PARAInstruction(Instruction):
    def __init__(self, data_name: str, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(PARAInstruction, self).__init__(data_name, data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            label = example["label"]
            example["target"] = self.verbalizer.get(str(label), '')
            example["verbalizer"] = SEP.join(list(set(self.verbalizer.values())))
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            example['start'] = self.get_start(example)
            assert example['instruction'][example['start']:example['start'] + len(example['target'])] == example['target']
            examples.append(example)
        return examples


class ClassificationInstruction(Instruction):
    def __init__(self, data_name: str, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(ClassificationInstruction, self).__init__(data_name, data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            label = example["label"]
            example["target"] = self.verbalizer.get(str(label), '')
            verbalizer = list(set(self.verbalizer.values()))
            # # 类别太多 长度过长，则sample 20个
            # if len(verbalizer) > 20:
            #     verbalizer = random.sample(verbalizer, 20)
            #     if example["target"] not in verbalizer:
            #         verbalizer.append(example["target"])
            #         verbalizer.sort()

            example["verbalizer"] = SEP.join(verbalizer)
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            example['start'] = self.get_start(example)

            a = example['instruction'][example['start']:example['start'] + len(example['target'])]
            assert a == example['target']
            examples.append(example)
        return examples


class WSCInstruction(Instruction):
    def __init__(self, data_name: str, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(WSCInstruction, self).__init__(data_name, data_list, verbalizer, instruction, keys_order, data_type)

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["verbalizer"] = SEP.join(list(set(self.verbalizer.values())))
            slots = [
                example["text"],
                example["target"]["span2_text"],
                example["target"]["span1_text"],
                example["verbalizer"],
            ]
            example["target"] = self.verbalizer.get(example["label"], '')
            example["instruction"] = self.instruction.format(*slots)
            example["data_type"] = self.data_type
            example['start'] = self.get_start(example)
            assert example['instruction'][example['start']:example['start'] + len(example['target'])] == example['target']
            examples.append(example)
        return examples


class WeiboEmotionInstruction(Instruction):
    def __init__(self, data_name: str, data_list: List, verbalizer: Dict, instruction: str, keys_order: List[str], data_type: str):
        super(WeiboEmotionInstruction, self).__init__(data_name, data_list, verbalizer, instruction, keys_order, data_type)
        self.verbalizer = self.verbalizer["label_list_1"]

    def transform2instruction(self):
        examples = []
        for sample in self.data_list:
            example = {k: v for k, v in sample.items()}
            example["target"] = self.verbalizer.get(example["label_1"], '')
            example["verbalizer"] = SEP.join(list(set(self.verbalizer.values())))
            example["instruction"] = self.instruction.format(*[
                example[k] for k in self.keys_order
            ])
            example["data_type"] = self.data_type
            example['start'] = self.get_start(example)
            assert example['instruction'][example['start']:example['start'] + len(example['target'])] == example['target']
            examples.append(example)
        return examples
