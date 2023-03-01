# -*- coding: utf-8 -*-
# @Time    : 2022/5/1 9:13 下午
# @Author  : JianingWang
# @File    : cpic_process
import json
import torch
import os.path
import numpy as np
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
from transformers import PreTrainedTokenizerBase
from processors.ProcessorBase import CLSProcessor
from metrics import datatype2metrics
from collections import defaultdict, Counter

from processors.benchmark.fewclue.utils import InputExample
from processors.instruction_prompting.chinese_extractive_instruction.instruction_templates import dataset2instruction
from processors.instruction_prompting.chinese_extractive_instruction.instruction_prompts import *
from processors.benchmark.fewclue.fewclue_processor import clue_processors, clue_output_modes, clue_task_to_instruction_type
from processors.benchmark.fewclue.data_collator import DataCollatorForGlobalPointer


# Used for mrc-based instruction-tuning for FewCLUE benchmark
class InstructionMRCForFewCLUEProcessor(CLSProcessor):
    def __init__(self,
                 data_args,
                 training_args,
                 model_args,
                 tokenizer=None,
                 post_tokenizer=False,
                 keep_raw_data=True):
        super().__init__(data_args,
                         training_args,
                         model_args,
                         tokenizer,
                         post_tokenizer=post_tokenizer,
                         keep_raw_data=keep_raw_data)
        param = {
            p.split('=')[0]: p.split('=')[1]
            for p in (data_args.user_defined).split(' ')
        }
        assert 'data_name' in param, "You must add one defined param 'data_name=xxx' in the user_defined parameter."
        self.data_name = param['data_name']
        self.is_pseudo = False  # 是否加载上一轮模型预测的dev和test的pseudo label
        self.pseudo_threshold = 1.0
        if 'is_pseudo' in param.keys():
            self.is_pseudo = bool(param['is_pseudo'])
            self.pseudo_threshold = float(param['pseudo_threshold'])
        self.data_dir = data_args.data_dir
        assert self.data_name in clue_processors.keys(
        ), 'Unknown task name {}'.format(self.data_name)
        self.processor = clue_processors[self.data_name]
        self.output_modes = clue_output_modes[self.data_name]
        self.train_file = os.path.join(data_args.data_dir,
                                       'train_few_all.json')  # 原始训练数据
        self.dev_file = os.path.join(data_args.data_dir, 'dev_few_all.json')
        self.test_file = os.path.join(data_args.data_dir,
                                      'test.json')  # fewclue榜单提交测试集
        self.test_file = os.path.join(data_args.data_dir,
                                      'test_public.json')  # fewclue公开测试集
        self.max_len = data_args.max_seq_length
        self.doc_stride = data_args.doc_stride
        self.sentence1_key = None
        self.labels = self.processor.get_labels()
        self.verbalizers = self.processor.get_verbalizers()

    def get_data_collator(self):
        pad_to_multiple_of_8 = self.training_args.fp16 and not self.data_args.pad_to_max_length
        return DataCollatorForGlobalPointer(
            self.tokenizer,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            pad_to_max_length=self.data_args.pad_to_max_length)

    def convert_to_instruction_template(self, examples: List[InputExample]):
        '''
        InputExample:
            self.guid = guid
            self.text_a = text_a
            self.text_b = text_b
            self.label = label
        '''
        # dict_examples = [example for example in examples]
        label_mappings = self.verbalizers
        # print("label_mappings=", label_mappings)
        # examples = [example for example in examples]
        # print("example=", examples[0])
        instruction_type = clue_task_to_instruction_type[self.data_name]
        format_info = dataset2instruction[instruction_type]
        instruction_processor = format_info['instruction'](
            self.data_name, examples, label_mappings, format_info['prompt'],
            format_info['keys_order'], format_info['data_type'])
        instruction_data = instruction_processor.transform2instruction()
        return instruction_data

    def get_examples(self, set_type):
        if set_type == 'train':
            examples = self.processor.get_train_examples(self.data_dir)
            instruction_data = self.convert_to_instruction_template(examples)
            examples = self._create_examples(instruction_data, 'train')
            examples = examples[:self.data_args.max_train_samples]
            self.train_examples = examples
        elif set_type == 'dev':
            examples = self.processor.get_dev_examples(self.data_dir)
            instruction_data = self.convert_to_instruction_template(examples)
            examples = self._create_examples(instruction_data, 'dev')
            examples = examples[:self.data_args.max_eval_samples]
            self.dev_examples = examples
        elif set_type == 'test':
            examples = self.processor.get_test_examples(self.data_dir)
            instruction_data = self.convert_to_instruction_template(examples)
            examples = self._create_examples(instruction_data, 'test')
            examples = examples[:self.data_args.max_predict_samples]
            self.test_examples = examples
        return examples

    def _create_examples(self, lines, set_type):
        examples = []
        is_train = 0 if set_type == 'test' else 1
        for line in lines:
            id_ = line['guid']  # 原始数据的编号
            text = line['instruction']  # 原始文本+候选+模板形成的最终输入序列
            target = line['target']  # 目标答案
            start = line['start']  # 目标答案在输入序列的起始位置
            data_type = line['data_type']  # 该任务的类型
            if data_type == 'ner':
                new_start, new_end = [], []
                for t, entity_starts in zip(target, start):
                    for s in entity_starts:
                        new_start.append(s)
                        new_end.append(s + len(t))
                start, end = new_start, new_end
                target = '|'.join(target)
            else:
                start, end = [start], [start + len(target)]

            examples.append({
                'id': id_,
                'content': text,
                'start': start,
                'end': end,
                'target': target,
                'data_type': data_type,
                'is_train': is_train
            })

        return examples

    def set_config(self, config):
        config.ent_type_size = 1
        config.inner_dim = 64
        config.RoPE = True

    def build_preprocess_function(self):
        # Tokenize the texts
        tokenizer = self.tokenizer
        max_seq_length = self.data_args.max_seq_length

        def func(examples):
            # Tokenize
            tokenized_examples = tokenizer(
                examples['content'],
                truncation=True,
                max_length=max_seq_length,
                padding='max_length'
                if self.data_args.pad_to_max_length else False,
                return_offsets_mapping=True)
            # 确定label
            return tokenized_examples

        return func

    def fush_multi_answer(self, has_answer, new_answer):
        # 对于某个id测试集，出现多个example时（例如同一个测试样本使用了多个模板而生成了多个example），此时将预测的topk结果进行合并
        # has为已经合并的结果，new为当前新产生的结果，
        # has格式为 {'ans': {'prob': float(prob[index_ids[ei]]), 'pos': (s, e)}, ...}
        # new {'ans': {'prob': float(prob[index_ids[ei]]), 'pos': (s, e)}, ...}
        # print('has_answer=', has_answer)
        for ans, value in new_answer.items():
            if ans not in has_answer.keys():
                has_answer[ans] = value
            else:
                has_answer[ans]['prob'] += value['prob']
                has_answer[ans]['pos'].extend(value['pos'])
        return has_answer

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
                    start_end = np.unravel_index(
                        entity, (self.data_args.max_seq_length,
                                 self.data_args.max_seq_length))
                    s = example['offset_mapping'][start_end[0]][0]
                    e = example['offset_mapping'][start_end[1]][1]
                    ans = example['content'][s:e]
                    if ans not in answer:
                        answer.append(ans)
                        # topk_answer.append({'answer': ans, 'prob': float(prob[index_ids[ei]]), 'pos': (s, e)})
                        topk_answer_dict[ans] = {
                            'prob': float(prob[index_ids[ei]]),
                            'pos': [(s, e)]
                        }

                predictions[id_] = answer
                if id_ not in topk_predictions.keys():
                    # print("topk_answer_dict=", topk_answer_dict)
                    topk_predictions[id_] = topk_answer_dict
                else:
                    # print("topk_predictions[id_]=", topk_predictions[id_])
                    topk_predictions[id_] = self.fush_multi_answer(
                        topk_predictions[id_], topk_answer_dict)
            else:
                best_start_end = np.unravel_index(
                    index[0], (self.data_args.max_seq_length,
                               self.data_args.max_seq_length))
                s = example['offset_mapping'][best_start_end[0]][0]
                e = example['offset_mapping'][best_start_end[1]][1]
                answer = example['content'][s:e]
                predictions[id_] = answer

                topk_answer_dict = dict()
                topk_index = index[prob > 0.0]
                index_ids = index_ids[prob > 0.0]
                # print('index_ids=', index_ids)
                for ei, index in enumerate(topk_index):
                    if ei > 6:
                        break
                    # 1D index转2D index
                    start_end = np.unravel_index(
                        index, (self.data_args.max_seq_length,
                                self.data_args.max_seq_length))
                    s = example['offset_mapping'][start_end[0]][0]
                    e = example['offset_mapping'][start_end[1]][1]
                    ans = example['content'][s:e]
                    # topk_answer.append({'answer': ans, 'prob': float(prob[index_ids[ei]]), 'pos': (s, e)})
                    topk_answer_dict[ans] = {
                        'prob': float(prob[index_ids[ei]]),
                        'pos': [(s, e)]
                    }

                predictions[id_] = answer
                if id_ not in topk_predictions.keys():
                    topk_predictions[id_] = topk_answer_dict
                else:
                    topk_predictions[id_] = self.fush_multi_answer(
                        topk_predictions[id_], topk_answer_dict)

        for id_, values in topk_predictions.items():
            # values {'ans': {}, ...}
            answer_list = list()
            for ans, value in values.items():
                answer_list.append({
                    'answer': ans,
                    'prob': value['prob'],
                    'pos': value['pos']
                })
            topk_predictions[id_] = answer_list

        return predictions, topk_predictions

    def compute_metrics(self, eval_predictions):
        examples = self.raw_datasets['validation']
        golden, dataname_map, dataname_type = {}, defaultdict(list), {}
        predictions, _ = self.get_predict_result(eval_predictions[0], examples)
        for example in examples:
            data_type = example['data_type']
            dataname = '_'.join(example['id'].split('_')[:-1])
            if dataname not in dataname_type:
                dataname_type[dataname] = data_type
            id_ = example['id']
            dataname_map[dataname].append(id_)
            if data_type == 'ner':
                golden[id_] = example['target'].split('|')
            else:
                golden[id_] = example['target']

        all_metrics = {
            'macro_f1': 0.,
            'micro_f1': 0.,
            'eval_num': 0,
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
            all_metrics['macro_f1'] += f1
            all_metrics['micro_f1'] += f1 * len(data_ids)
            all_metrics['eval_num'] += len(data_ids)
            all_metrics[dataname] = round(acc, 4)
        all_metrics['macro_f1'] = round(
            all_metrics['macro_f1'] / len(dataname_map), 4)
        all_metrics['micro_f1'] = round(
            all_metrics['micro_f1'] / all_metrics['eval_num'], 4)
        return all_metrics

    def save_result(self, logits, label_ids):
        examples = self.raw_datasets['test']
        predicts, topk_predicts = self.get_predict_result(logits, examples)
        # print('topk_predicts=', topk_predicts)
        verbalizer_ = {v: k for k, v in self.verbalizers.items()}
        labels = self.labels
        submit_predicts = dict()
        for key, value in predicts.items():
            submit_predicts[int(key.split('-')[1])] = labels[int(
                verbalizer_[value])]

        outfile = os.path.join(self.training_args.output_dir, 'answer.json')
        with open(outfile, 'w', encoding='utf8') as f:
            json.dump(submit_predicts, f, ensure_ascii=False, indent=2)

        topk_file = os.path.join(self.training_args.output_dir,
                                 'topk_prob.json')
        with open(topk_file, 'w', encoding='utf8') as f2:
            json.dump(topk_predicts, f2, ensure_ascii=False, indent=2)

    def create_test_label_data(self,
                               examples,
                               out,
                               pos,
                               tag: dict = None,
                               threshole=0.9):
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
            best_pos = pos[k][best_result]  # (x, x) or [(x, x), ..]
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
                    if content[pos[0]:pos[1]] != target:
                        continue
                    example['start'] = [pos[0]]
                    example['end'] = [pos[1]]
                    example['target'] = target
                    new_example.append(example)
                else:
                    assert type(pos) == list and type(pos[0]) == list and type(
                        pos[0][0]) == int
                    for pos_i in pos:
                        if content[pos_i[0]:pos_i[1]] == target:
                            example['start'] = [pos_i[0]]
                            example['end'] = [pos_i[1]]
                            example['target'] = target
                            new_example.append(example)
                            break

        print('example ==')
        print(new_example[0])
        print('correct answer num: {}'.format(len(new_example)))
        return new_example
