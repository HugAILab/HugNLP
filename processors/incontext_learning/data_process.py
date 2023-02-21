# -*- coding: utf-8 -*-
# @Time    : 2022/9/8 9:13 下午
# @Author  : JianingWang
# @File    : data_process
import enum
import json
# from random import random
import random
import torch
import os.path
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from collections import defaultdict
from transformers import PreTrainedTokenizerBase
from processors.benchmark.clue.utils import InputExample

from processors.benchmark.glue.glue_processor import glue_processors, compute_metrics_mapping
from processors.ProcessorBase import CLSProcessor
from metrics import datatype2metrics
from collections import defaultdict, Counter


all_processors = [glue_processors]
all_processors = {task_name: task_processor for processor in all_processors for task_name, task_processor in processor.items()}

@dataclass
class InContextLearningCollatorForGPT:
    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    pad_to_max_length: Optional[bool] = None

    def __call__(self, features):
        # Tokenize
        is_train = features[0]['is_train'] > 0
        batch = []
        for f in features:
            input_dict = {'id': f['id'],
                        'input_ids': f['input_ids'],
                        'token_type_ids': f['token_type_ids'],
                        'attention_mask': f['attention_mask'],
                        'labels': f['input_ids'],
                        }
            batch.append(input_dict)
        '''
        batch['input_ids'].shape = [n, len]
        '''
        batch = self.tokenizer.pad(
            batch,
            padding='max_length',  # 为了index不出错直接Padding到max length，如果用longest，后面的np.unravel_index也要改
            max_length=self.max_length,
            return_tensors="pt"
        )
        # 将所有id相同的放在一起
        option_size = len(f["options"][0])
        assert len(batch["input_ids"]) % option_size == 0
        new_batch = {"input_ids": list(), "token_type_ids": list(), "attention_mask": list()}
        for i in range(0, len(batch["input_ids"]), option_size):
            new_batch["input_ids"].append(batch["input_ids"][i: i + option_size])
            new_batch["token_type_ids"].append(batch["token_type_ids"][i: i + option_size])
            new_batch["attention_mask"].append(batch["attention_mask"][i: i + option_size])
            new_batch["labels"].append(batch["labels"][i: i + option_size])

        new_batch["input_ids"] = torch.stack(new_batch["input_ids"])
        new_batch["token_type_ids"] = torch.stack(new_batch["token_type_ids"])
        new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
        new_batch["labels"] = torch.stack(new_batch["labels"])
        batch["options"] = torch.Tensor([list(range(len(f['options']))) for f in features]).long()
        # new_batch["input_ids"].shape = [n, option_size, len]
        return new_batch

# GPT for in-context learning
class InContextLearningProcessor(CLSProcessor):
    def __init__(self, data_args, training_args, model_args, tokenizer=None, post_tokenizer=False, keep_raw_data=True):
        super().__init__(data_args, training_args, model_args, tokenizer, post_tokenizer=post_tokenizer, keep_raw_data=keep_raw_data)
        param = {p.split("=")[0]: p.split("=")[1] for p in (data_args.user_defined).split(" ")}
        assert "data_name" in param, "You must add one defined param 'data_name=xxx' in the user_defined parameter."
        self.data_name = param["data_name"] # task-specific name
        self.num_incontext_example = int(param["num_incontext_example"])
        self.task_format = param["task_format"] # classification / qa / generation / probing
        self.data_dir = data_args.data_dir
        self.train_file = os.path.join(data_args.data_dir, 'train.csv') # 原始训练数据
        self.dev_file = os.path.join(data_args.data_dir, 'dev.csv')
        self.test_file = os.path.join(data_args.data_dir, 'test.csv')
        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.sentence1_key = None
        self.processor = [glue_processors] # 多个benchmark的结合
        self.processor = all_processors[self.data_name]
        self.labels = self.processor.get_labels()
        # prompt related
        self.verbalizers = self.processor.get_verbalizers() # label word mapping
        self.options = [v[0] if type(v) == list else v for k, v in self.verbalizers] # label word list
        self.prompt_prefix = param["prompt_prefix"] if "prompt_prefix" in param.keys() else "" # prefix template. e.g. Asnwer next question.
        self.q_prefix = param["q_prefix"] if "q_prefix" in param.keys() else "Input: " # question predix template. e.g. Input:
        self.a_prefix = param["a_prefix"] if "a_prefix" in param.keys() else "Output: " # answer prefix template. e.g. Output:
        # 确保dev和test使用的是相同的in-context example，因此在创建processor的同时，采样好对应的in-context example
        self.incontext_examples = self.InContextSampling(self.get_examples("train"))
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token # GPT需要显式地添加padding token

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return InContextLearningCollatorForGPT(self.tokenizer, pad_to_multiple_of=8 if pad_to_multiple_of_8 else None, pad_to_max_length=self.data_args.pad_to_max_length)

    def InContextSampling(self, examples: List[InputExample]) -> List[InputExample]:
        # used for sampling in-context examples
        random.seed = self.data_args.seed
        random.shuffle(examples)
        incontext_examples = examples[:self.num_incontext_example] if self.num_incontext_example > len(examples) else examples
        return incontext_examples
    
    def construct_prompt(self, incontext_examples: List[InputExample], eval_example: InputExample):
        '''
        generate prompt with multiple in-context examples.
        prompt_prefix = "What are follows emotions?"
        q_prefix = "Input: "
        a_prefix = "Output: " 
        prompt = "What are follows emotions? Input: The book is very nice.\n Output: great.\n\n Input: I never eat chocolate!\n Output: bad.\n\n Input: This film is wonderful.\n Output: "
        '''
        prompt = self.prompt_prefix
        for incontext_example in incontext_examples:
            s = incontext_example["text_a"] + (" " + incontext_example["text_b"]) if incontext_example["text_b"] is not None else ""
            l = incontext_example["label"]
            if self.task_format == "classification":
                # 需要将类别标签转换为词
                label_word = self.verbalizers[l][0] if isinstance(self.verbalizers[l], list) else self.verbalizers[l]
            else:
                label_word = l
            prompt += self.q_prefix
            prompt += s + "\n"
            prompt += self.a_prefix
            prompt += label_word + "\n\n"

        eval_s = eval_example["text_a"] + (" " + eval_example["text_b"]) if eval_example["text_b"] is not None else ""
        label = eval_example["label"]
        prompt += self.q_prefix
        prompt += eval_s + "\n"
        assert self.a_prefix[-1] == " "
        prompt += self.a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
        return prompt, label

    def get_examples(self, set_type):
        # assert set_type != "train", "In-context learning dose not have training proce"
        if set_type == "train":
            # 仅用于支持本框架（默认必须加载训练集）
            examples = self.processor.get_train_examples(self.data_dir)
            return examples # List[InputExample]
        else:# dev或test时
            # 先获取所有的训练集
            training_examples = self.processor.get_train_examples(self.data_dir)
            # 随机采样若干in-context example作为demonstration
            # incontext_examples = self.InContextSampling(training_examples)
            incontext_examples = self.incontext_examples
            if set_type == "dev":
                examples: List[InputExample] = self.processor.get_dev_examples(self.data_dir)
            elif set_type == "test":
                examples: List[InputExample] = self.processor.get_test_examples(self.data_dir)
            # 为每个dev/test构建prompt 
            eval_examples = list()
            for example in examples:
                prompt, label = self.construct_prompt(incontext_examples, example)
                eval_examples.append(InputExample(guid=example.guid, text_a=prompt, text_b=None, label=label))
            examples = self._create_examples(eval_examples, set_type=set_type)
        return examples # List[dict]
    
    def _create_examples(self, lines, set_type=None):
        examples = []
        is_train = 0 if set_type == 'test' else 1
        for ei, line in enumerate(lines):
            id_ = ei
            prompt = line["text_a"]
            label = line["label"]
            if self.task_format == "classification":
                for option in self.options:
                    examples.append({'id': id_, # 测试样本编号
                                    'prompt': prompt.strip() + " " + option, # 每个候选label对应一个prompt
                                    'options': self.options, # 所有候选label
                                    'target': label, # 当前测试样本正确的label
                                    'data_type': self.task_format, # 任务类型
                                    'is_train': is_train
                                    })
            else:
                examples.append({'id': id_,
                                'prompt': prompt, # 原始prompt
                                'label': label,
                                'data_type': self.task_format,
                                'is_train': is_train
                                })
        return examples

    def set_config(self, config):
        pass

    def build_preprocess_function(self):
        # Tokenize the texts
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length

        def func(examples):
            # Tokenize
            tokenized_examples = tokenizer(
                examples['prompt'],
                truncation=True,
                max_length=max_seq_length,
                padding="max_length" if self.data_args.pad_to_max_length else False,
                # return_offsets_mapping=True
            )
            # 确定label
            return tokenized_examples

        return func


    def get_predict_result(self, logits, examples):
        probs, indices = logits
        probs = probs.squeeze(1)  # topk结果的概率
        indices = indices.squeeze(1)  # topk结果的索引
        # print('probs=', probs) # [n, m]
        # print('indices=', indices) # [n, m]
        predictions = {}
        topk_predictions = {}
        for prob, index, example in zip(probs, indices, examples):
            data_type = example['data_type']
            id_ = example['id']
            index_ids = torch.Tensor([i for i in range(len(index))]).long()
            topk_answer = list()
            if data_type == 'ner':
                answer = []
                topk_answer_dict = dict()
                # TODO 1. 调节阈值 2. 处理输出实体重叠问题
                entity_index = index[prob > 0.0]
                index_ids = index_ids[prob > 0.0]
                for ei, entity in enumerate(entity_index):
                    # 1D index转2D index
                    start_end = np.unravel_index(entity, (self.data_args.max_seq_length, self.data_args.max_seq_length))
                    s = example['offset_mapping'][start_end[0]][0]
                    e = example['offset_mapping'][start_end[1]][1]
                    ans = example['content'][s: e]
                    if ans not in answer:
                        answer.append(ans)
                        # topk_answer.append({'answer': ans, 'prob': float(prob[index_ids[ei]]), 'pos': (s, e)})
                        topk_answer_dict[ans] = {'prob': float(prob[index_ids[ei]]), 'pos': [(s, e)]}

                predictions[id_] = answer
                if id_ not in topk_predictions.keys():
                    # print("topk_answer_dict=", topk_answer_dict)
                    topk_predictions[id_] = topk_answer_dict
                else:
                    # print("topk_predictions[id_]=", topk_predictions[id_])
                    topk_predictions[id_] = self.fush_multi_answer(topk_predictions[id_], topk_answer_dict)
            else:
                best_start_end = np.unravel_index(index[0], (self.data_args.max_seq_length, self.data_args.max_seq_length))
                s = example['offset_mapping'][best_start_end[0]][0]
                e = example['offset_mapping'][best_start_end[1]][1]
                answer = example['content'][s: e]
                predictions[id_] = answer

                topk_answer_dict = dict()
                topk_index = index[prob > 0.0]
                index_ids = index_ids[prob > 0.0]
                # print('index_ids=', index_ids)
                for ei, index in enumerate(topk_index):
                    if ei > 6:
                        break
                    # 1D index转2D index
                    start_end = np.unravel_index(index, (self.data_args.max_seq_length, self.data_args.max_seq_length))
                    s = example['offset_mapping'][start_end[0]][0]
                    e = example['offset_mapping'][start_end[1]][1]
                    ans = example['content'][s: e]
                    # topk_answer.append({'answer': ans, 'prob': float(prob[index_ids[ei]]), 'pos': (s, e)})
                    topk_answer_dict[ans] = {'prob': float(prob[index_ids[ei]]), 'pos': [(s, e)]}

                predictions[id_] = answer
                if id_ not in topk_predictions.keys():
                    topk_predictions[id_] = topk_answer_dict
                else:
                    topk_predictions[id_] = self.fush_multi_answer(topk_predictions[id_], topk_answer_dict)

        for id_, values in topk_predictions.items():
            # values {'ans': {}, ...}
            answer_list = list()
            for ans, value in values.items():
                answer_list.append({'answer': ans, 'prob': value['prob'], 'pos': value['pos']})
            topk_predictions[id_] = answer_list

        return predictions, topk_predictions

    def compute_metrics(self, eval_predictions):
        examples = self.raw_datasets['validation']
        golden, dataname_map, dataname_type = {}, defaultdict(list), {}
        predictions, _ = self.get_predict_result(eval_predictions[0], examples)
        for example in examples:
            data_type = example['data_type']
            dataname = "_".join(example["id"].split("_")[:-1])
            if dataname not in dataname_type:
                dataname_type[dataname] = data_type
            id_ = example['id']
            dataname_map[dataname].append(id_)
            if data_type == 'ner':
                golden[id_] = example['target'].split('|')
            else:
                golden[id_] = example['target']

        all_metrics = {
            "macro_f1": 0.,
            "micro_f1": 0.,
            "eval_num": 0,
        }

        for dataname, data_ids in dataname_map.items():
            metric = datatype2metrics[dataname_type[dataname]]()
            gold = {k: v for k, v in golden.items() if k in data_ids}
            pred = {k: v for k, v in predictions.items() if k in data_ids}
            score = metric.calc_metric(golden=gold, predictions=pred)
            # print('score=', score)
            acc, f1 = score['acc'], score['f1']
            # if len(gold) != len(pred) or len(gold) < 20:
                # print(dataname, dataname_type[dataname], round(acc, 4), len(gold), len(pred), data_ids)
            all_metrics["macro_f1"] += f1
            all_metrics["micro_f1"] += f1 * len(data_ids)
            all_metrics["eval_num"] += len(data_ids)
            all_metrics[dataname] = round(acc, 4)
        all_metrics["macro_f1"] = round(all_metrics["macro_f1"] / len(dataname_map), 4)
        all_metrics["micro_f1"] = round(all_metrics["micro_f1"] / all_metrics["eval_num"], 4)
        return all_metrics

    def save_result(self, logits, label_ids):
        examples = self.raw_datasets['test']
        predicts, topk_predicts = self.get_predict_result(logits, examples)
        # print('topk_predicts=', topk_predicts)

        outfile = os.path.join(self.training_args.output_dir, 'answer.json')
        with open(outfile, 'w', encoding='utf8') as f:
            json.dump(predicts, f, ensure_ascii=False, indent=2)

        topk_file = os.path.join(self.training_args.output_dir, 'topk_prob.json')
        with open(topk_file, 'w', encoding='utf8') as f2:
            json.dump(topk_predicts, f2, ensure_ascii=False, indent=2)


    def create_test_label_data(self, examples, out, pos, tag: dict=None, threshole=0.9):
        '''
        该函数用于生成dev数据集
        out: 每个样本对应的Topk个预测结果及其得分
        {"InsuranceIntentChange_TEST_95": {
            "变更车辆信息": 4.9753875732421875,
            "客户信息变更": 1.5599589943885803,
            "变更车辆信息‖客户信息变更": 0.11198210716247559,
          },...
        }
        '''


        # 构建映射
        '''
        examples {'id': id_,
         'content': text,
         'start': start,
         'end': end,
         'target': target,
         'data_type': data_type,
         'is_train': is_train})
        '''
        model_num = 6
        template_per_model_num = 1
        correct_answer = dict() 
        for k, v in out.items():
            if 'ner' in k.lower():
                continue
            v = sorted(v.items(), key=lambda x: x[1], reverse=True)
            best_result, best_prob = v[0][0], v[0][1]
            best_pos = pos[k][best_result] # (x, x) or [(x, x), ..]
            if best_prob >= threshole * model_num * template_per_model_num:
                correct_answer[k] = (best_pos, best_result)
        # if tag is not None:
        #     for key, value in tag.items():
        #         correct_answer[key] = value
        # 构建dev数据集
        new_example = list()
        for example in examples:
            id = example['id']
            # print('id=', id)
            if id in correct_answer.keys():
                content = example['content']
                target = correct_answer[id][1]
                pos = correct_answer[id][0]
                if type(pos[0]) == int:
                    if content[pos[0]: pos[1]] != target:
                        continue
                    example['start'] = [pos[0]]
                    example['end'] = [pos[1]]
                    example['target'] = target
                    new_example.append(example)
                else:
                    assert type(pos) == list and type(pos[0]) == list and type(pos[0][0]) == int
                    for pos_i in pos:
                        if content[pos_i[0]: pos_i[1]] == target:
                            example['start'] = [pos_i[0]]
                            example['end'] = [pos_i[1]]
                            example['target'] = target
                            new_example.append(example)
                            break

        print("example ==")
        print(new_example[0])
        print("correct answer num: {}".format(len(new_example)))
        return new_example

# Used for generation-based instruction-tuning in Chinese
class ChineseInstructionGenProcessor(CLSProcessor):
    pass