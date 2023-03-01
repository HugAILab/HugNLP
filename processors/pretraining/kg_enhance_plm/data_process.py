# -*- coding: utf-8 -*-
# @Time    : 2021/12/2 4:49 下午
# @Author  : JianingWang(ruihan.wjn)
# @File    : MLMProcessor
import torch
import logging
from itertools import chain
from typing import List, Union, Dict, Any, Optional, Tuple
import numpy as np
import random
import warnings
from datasets import load_dataset
from transformers import EvalPrediction, BatchEncoding, BertTokenizer, BertTokenizerFast, PreTrainedTokenizerBase
from processors.ProcessorBase import DataProcessor
from datasets import Dataset, load_from_disk
from data.data_collator import DataCollatorForLanguageModeling
from tools.processing_utils.common import is_chinese_char, is_chinese
from processors.pretraining.kg_enhance_plm.kg_prompt import KGPrompt

logger = logging.getLogger(__name__)


class DataCollatorForProcessedWikiKGPLM(DataCollatorForLanguageModeling):
    """Data collator used for language modeling that masks entire words.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    <Tip>

    This collator relies on details of the implementation of subword tokenization by [`BertTokenizer`], specifically
    that subword tokens are prefixed with *##*. For tokenizers that do not adhere to this scheme, this collator will
    produce an output that is roughly equivalent to [`.DataCollatorForLanguageModeling`].

    </Tip>
    """
    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                'This tokenizer does not have a mask token which is necessary for masked language modeling. '
                'You should pass `mlm=False` to train on causal language modeling instead.'
            )
        self.numerical_tokens = [
            v for k, v in self.tokenizer.vocab.items() if k.isdigit()
        ]
        self.exclude_tokens = self.numerical_tokens + self.tokenizer.all_special_ids

        # add by wjn
        self.kg_prompt = KGPrompt(tokenizer=self.tokenizer)
        random.seed(42)

    def torch_call(
        self, features: List[Union[List[int], Any,
                                   Dict[str, Any]]]) -> Dict[str, Any]:
        assert isinstance(features[0], (dict, BatchEncoding))
        '''
        task = 1:
        {
            'input_ids': input_ids,
            'kg_prompt_ids': kg_prompt_ids,
            'noise_detect_label': prompt['noise_detect_label'],
            'task_id': task,
        }

        task = 2:
        {
            'input_ids': input_ids,
            'kg_prompt_ids': kg_prompt_ids,
            'entity_label': entity_label,
            'entity_negative': entity_negative,
            'task_id': task,
        }

        task = 3:
        {
            'input_ids': input_ids,
            'kg_prompt_ids': kg_prompt_ids,
            'relation_label': relation_label,
            'relation_negative': relation_negative,
            'task_id': task,
        }
        '''
        '''
        为当前整个batch设定任务类型，以一定概率生成不同任务的数据
        task=1：Masked Language Modeling任务，只对context（token_type_ids=1）部分随机mask
        task=2：Entity Completion：随机只mask一个实体
        task=3：Relation Classification：随机只mask一个关系
        '''
        # start_time = time()

        input_features = list()

        for ei, feature in enumerate(features):

            input_ids, kg_prompt_ids, task_id = feature['input_ids'], feature[
                'kg_prompt_ids'], feature['task_id']
            text_len = len(input_ids)
            kg_len = len(kg_prompt_ids)
            # 补充padding
            input_ids = input_ids + kg_prompt_ids
            input_ids = input_ids[:self.tokenizer.model_max_length]
            token_len = len(input_ids)
            input_ids.extend(
                [self.tokenizer.pad_token_id] *
                (self.tokenizer.model_max_length - len(input_ids)))
            # 生成token_type_id
            token_type_ids = [0] * text_len + [0] * kg_len + [0] * (
                self.tokenizer.model_max_length - text_len - kg_len)
            # 生成mlm_label
            mlm_labels = [-100] * len(input_ids)

            attention_mask = np.zeros([
                self.tokenizer.model_max_length,
                self.tokenizer.model_max_length
            ])
            # # 非pad部分，context全部可见
            token_type_span = [(0, text_len)]
            st, ed = text_len, text_len
            for ei, token in enumerate(kg_prompt_ids):
                if token == self.tokenizer.sep_token_id:
                    ed = text_len + ei
                    token_type_span.append((st, ed))
                    st = ei
            context_start, context_end = token_type_span[0]
            attention_mask[context_start:context_end, :token_len] = 1
            attention_mask[:token_len, context_start:context_end] = 1
            # 非pad部分，每个三元组自身可见、与context可见，三元组之间不可见
            for ei, (start, end) in enumerate(token_type_span):
                # start, end 每个token type的区间
                attention_mask[start:end, start:end] = 1
            attention_mask = attention_mask.tolist()

            # 由于原始不同的task对应的feature不同，为了统一，对不存在的feature使用pad进行填充
            # entity_label = [0] * 20
            # entity_negative = [[0] * 20] * 5
            # relation_label = [0] * 5
            # relation_negative = [[0] * 5] * 5
            entity_candidate = [[0] * 20] * 6
            relation_candidate = [[0] * 5] * 6

            if task_id == 1:
                input_ids = torch.Tensor(input_ids).long()
                mlm_labels = input_ids.clone()
                # MLM mask采样
                # 只对context部分进行mlm采样。15%的进行mask，其中80%替换为<MASK>，10%随机替换其他词，10%保持不变
                probability_matrix = torch.full([text_len],
                                                self.mlm_probability)
                probability_matrix = torch.cat([
                    probability_matrix,
                    torch.zeros([self.tokenizer.model_max_length - text_len])
                ], -1)

                masked_indices = torch.bernoulli(probability_matrix).bool()
                mlm_labels[
                    ~masked_indices] = -100  # We only compute loss on masked tokens

                # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
                indices_replaced = torch.bernoulli(
                    torch.full(mlm_labels.shape, 0.8)).bool() & masked_indices
                input_ids[
                    indices_replaced] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.mask_token)

                # 10% of the time, we replace masked input tokens with random word
                indices_random = torch.bernoulli(
                    torch.full(
                        mlm_labels.shape,
                        0.5)).bool() & masked_indices & ~indices_replaced
                random_words = torch.randint(len(self.tokenizer.get_vocab()),
                                             mlm_labels.shape,
                                             dtype=torch.long)
                input_ids[indices_random] = random_words[indices_random]

                # for ei, token_id in enumerate(input_ids[1:]):
                #     if token_type_ids[ei] != 0 or token_id == self.tokenizer.pad_token_id:
                #         break
                #     if token_id < 10 or token_id == self.tokenizer.sep_token_id:
                #         continue
                #     if random.random() <= 0.15:
                #         mlm_labels[ei] = token_id
                #         rd = random.random()
                #         if rd <= 0.8:
                #             input_ids[ei] = self.tokenizer.mask_token_id
                #         elif rd > 0.9:
                #             input_ids[ei] = random.randint(10, len(self.tokenizer.get_vocab()) - 1)
                input_features.append({
                    'input_ids':
                    input_ids.numpy().tolist(),
                    'token_type_ids':
                    token_type_ids,
                    'attention_mask':
                    attention_mask,
                    'labels':
                    mlm_labels.numpy().tolist(),
                    # 'entity_label': entity_label,
                    'entity_candidate':
                    entity_candidate,
                    # 'relation_label': relation_label,
                    'relation_candidate':
                    relation_candidate,
                    'noise_detect_label':
                    feature['noise_detect_label'],
                    'task_id':
                    task_id,
                    'mask_id':
                    self.tokenizer.mask_token_id
                })

            elif task_id == 2:
                entity_label, entity_negative = feature[
                    'entity_label'], feature['entity_negative']
                label_index = 0
                for ei, token in enumerate(kg_prompt_ids):
                    if token == self.tokenizer.mask_token_id:
                        label_index += 1
                        mlm_labels[text_len + ei] = entity_label[label_index]

                entity_candidate = entity_negative + [entity_label]

                input_features.append({
                    'input_ids': input_ids,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask,
                    'labels': mlm_labels,
                    # 'entity_label': entity_label,
                    'entity_candidate': entity_candidate,
                    # 'relation_label': relation_label,
                    'relation_candidate': relation_candidate,
                    'noise_detect_label': -1,
                    'task_id': task_id,
                    'mask_id': self.tokenizer.mask_token_id
                })

            elif task_id == 3:
                relation_label, relation_negative = feature[
                    'relation_label'], feature['relation_negative']
                label_index = 0
                for ei, token in enumerate(kg_prompt_ids):
                    if token == self.tokenizer.mask_token_id:
                        label_index += 1
                        mlm_labels[text_len + ei] = relation_label[label_index]

                relation_candidate = relation_negative + [relation_label]

                input_features.append({
                    'input_ids': input_ids,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask,
                    'labels': mlm_labels,
                    # 'entity_label': entity_label,
                    'entity_candidate': entity_candidate,
                    # 'relation_label': relation_label,
                    'relation_candidate': relation_candidate,
                    'noise_detect_label': -1,
                    'task_id': task_id,
                    'mask_id': self.tokenizer.mask_token_id
                })
        del features
        # print("============= feature demo: =============")
        # print("input_ids=", input_features[0]['input_ids'])
        # print("token_type_ids=", input_features[0]['token_type_ids'])
        # print("mlm_labels=", input_features[0]['labels'])
        # print("task_id=", input_features[0]['task_id'])
        # print("entity_candidate=", input_features[0]['entity_candidate'])
        # print("relation_candidate=", input_features[0]['relation_candidate'])

        input_features = {
            key: torch.tensor([feature[key] for feature in input_features],
                              dtype=torch.long)
            for key in input_features[0].keys()
        }

        # end_time = time()
        # print('data_process_time: {}'.format(end_time - start_time))
        return input_features

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """Get 0/1 labels for masked tokens with whole word mask proxy."""
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                'DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. '
                'Please refer to the documentation for more information.')

        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == '[CLS]' or token == '[SEP]':
                continue

            if len(cand_indexes) >= 1 and token.startswith('##'):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(
            max_predictions,
            max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError(
                'Length of covered_indexes is not equal to length of masked_lms.'
            )
        mask_labels = [
            1 if i in covered_indexes else 0 for i in range(len(input_tokens))
        ]
        return mask_labels

    def torch_mask_tokens(
            self,
            inputs: Any,
            special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
        import torch

        labels = inputs.clone()  # [bz, len]
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(
            labels.shape,
            self.mlm_probability)  # [bz, seq_len], all values is 0.15
        special_tokens_mask = [[
            1 if token in self.exclude_tokens else 0 for token in val
        ] for val in labels.tolist()]
        special_tokens_mask = torch.tensor(special_tokens_mask,
                                           dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        ### add by ruihan.wjn relace all values to 0 where belongs to kg_prompt

        ###
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer),
                                     labels.shape,
                                     dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class DataCollatorForProcessedWikiKGPLM_OnlyMLM(DataCollatorForLanguageModeling
                                                ):
    """Data collator used for language modeling that masks entire words. Only MLM.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    <Tip>

    This collator relies on details of the implementation of subword tokenization by [`BertTokenizer`], specifically
    that subword tokens are prefixed with *##*. For tokenizers that do not adhere to this scheme, this collator will
    produce an output that is roughly equivalent to [`.DataCollatorForLanguageModeling`].

    </Tip>
    """
    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                'This tokenizer does not have a mask token which is necessary for masked language modeling. '
                'You should pass `mlm=False` to train on causal language modeling instead.'
            )
        self.numerical_tokens = [
            v for k, v in self.tokenizer.vocab.items() if k.isdigit()
        ]
        self.exclude_tokens = self.numerical_tokens + self.tokenizer.all_special_ids

        # add by wjn
        self.kg_prompt = KGPrompt(tokenizer=self.tokenizer)
        random.seed(42)

    def torch_call(
        self, features: List[Union[List[int], Any,
                                   Dict[str, Any]]]) -> Dict[str, Any]:
        assert isinstance(features[0], (dict, BatchEncoding))
        '''
        task = 1:
        {
            'input_ids': input_ids,
            'kg_prompt_ids': kg_prompt_ids,
            'noise_detect_label': prompt['noise_detect_label'],
            'task_id': task,
        }

        task = 2:
        {
            'input_ids': input_ids,
            'kg_prompt_ids': kg_prompt_ids,
            'entity_label': entity_label,
            'entity_negative': entity_negative,
            'task_id': task,
        }

        task = 3:
        {
            'input_ids': input_ids,
            'kg_prompt_ids': kg_prompt_ids,
            'relation_label': relation_label,
            'relation_negative': relation_negative,
            'task_id': task,
        }
        '''
        '''
        为当前整个batch设定任务类型，以一定概率生成不同任务的数据
        task=1：Masked Language Modeling任务，只对context（token_type_ids=1）部分随机mask
        task=2：Entity Completion：随机只mask一个实体
        task=3：Relation Classification：随机只mask一个关系
        '''
        # start_time = time()

        input_features = list()

        for ei, feature in enumerate(features):

            input_ids, kg_prompt_ids, task_id = feature['input_ids'], feature[
                'kg_prompt_ids'], feature['task_id']
            text_len = len(input_ids)
            kg_len = len(kg_prompt_ids)
            # 补充padding
            input_ids = input_ids + kg_prompt_ids
            input_ids = input_ids[:self.tokenizer.model_max_length]
            token_len = len(input_ids)
            input_ids.extend(
                [self.tokenizer.pad_token_id] *
                (self.tokenizer.model_max_length - len(input_ids)))
            # 生成token_type_id
            token_type_ids = [0] * text_len + [0] * kg_len + [0] * (
                self.tokenizer.model_max_length - text_len - kg_len)
            # 生成mlm_label
            # mlm_labels = [-100] * len(input_ids)

            attention_mask = np.zeros([
                self.tokenizer.model_max_length,
                self.tokenizer.model_max_length
            ])
            # # 非pad部分，context全部可见
            token_type_span = [(0, text_len)]
            st, ed = text_len, text_len
            for ei, token in enumerate(kg_prompt_ids):
                if token == self.tokenizer.sep_token_id:
                    ed = text_len + ei
                    token_type_span.append((st, ed))
                    st = ei
            context_start, context_end = token_type_span[0]
            attention_mask[context_start:context_end, :token_len] = 1
            attention_mask[:token_len, context_start:context_end] = 1
            # 非pad部分，每个三元组自身可见、与context可见，三元组之间不可见
            for ei, (start, end) in enumerate(token_type_span):
                # start, end 每个token type的区间
                attention_mask[start:end, start:end] = 1
            attention_mask = attention_mask.tolist()

            # 由于原始不同的task对应的feature不同，为了统一，对不存在的feature使用pad进行填充
            # entity_label = [0] * 20
            # entity_negative = [[0] * 20] * 5
            # relation_label = [0] * 5
            # relation_negative = [[0] * 5] * 5
            entity_candidate = [[0] * 20] * 6
            relation_candidate = [[0] * 5] * 6

            # MLM mask采样
            # 只对context部分进行mlm采样。15%的进行mask，其中80%替换为<MASK>，10%随机替换其他词，10%保持不变

            input_ids = torch.Tensor(input_ids).long()
            mlm_labels = input_ids.clone()
            # MLM mask采样
            # 只对context部分进行mlm采样。15%的进行mask，其中80%替换为<MASK>，10%随机替换其他词，10%保持不变
            probability_matrix = torch.full([text_len], self.mlm_probability)
            probability_matrix = torch.cat([
                probability_matrix,
                torch.zeros([self.tokenizer.model_max_length - text_len])
            ], -1)

            masked_indices = torch.bernoulli(probability_matrix).bool()
            mlm_labels[
                ~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(
                torch.full(mlm_labels.shape, 0.8)).bool() & masked_indices
            input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(
                torch.full(mlm_labels.shape,
                           0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer.get_vocab()),
                                         mlm_labels.shape,
                                         dtype=torch.long)
            input_ids[indices_random] = random_words[indices_random]

            # for ei, token_id in enumerate(input_ids[1:]):
            #     if token_type_ids[ei] != 0 or token_id == self.tokenizer.pad_token_id:
            #         break
            #     if token_id < 10 or token_id == self.tokenizer.sep_token_id:
            #         continue
            #     if random.random() <= 0.15:
            #         mlm_labels[ei] = token_id
            #         rd = random.random()
            #         if rd <= 0.8:
            #             input_ids[ei] = self.tokenizer.mask_token_id
            #         elif rd > 0.9:
            #             input_ids[ei] = random.randint(10, len(self.tokenizer.get_vocab()) - 1)
            input_features.append({
                'input_ids': input_ids.numpy().tolist(),
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels': mlm_labels.numpy().tolist(),
                # 'entity_label': entity_label,
                'entity_candidate': entity_candidate,
                # 'relation_label': relation_label,
                'relation_candidate': relation_candidate,
                # 'noise_detect_label': -1,
                'task_id': task_id,
                'mask_id': self.tokenizer.mask_token_id
            })

        del features
        # print("============= feature demo: =============")
        # print("input_ids=", input_features[0]['input_ids'])
        # print("token_type_ids=", input_features[0]['token_type_ids'])
        # print("mlm_labels=", input_features[0]['labels'])
        # print("task_id=", input_features[0]['task_id'])
        # print("entity_candidate=", input_features[0]['entity_candidate'])
        # print("relation_candidate=", input_features[0]['relation_candidate'])

        input_features = {
            key: torch.tensor([feature[key] for feature in input_features],
                              dtype=torch.long)
            for key in input_features[0].keys()
        }

        # end_time = time()
        # print('data_process_time: {}'.format(end_time - start_time))
        return input_features

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """Get 0/1 labels for masked tokens with whole word mask proxy."""
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                'DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. '
                'Please refer to the documentation for more information.')

        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token == '[CLS]' or token == '[SEP]':
                continue

            if len(cand_indexes) >= 1 and token.startswith('##'):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(
            max_predictions,
            max(1, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError(
                'Length of covered_indexes is not equal to length of masked_lms.'
            )
        mask_labels = [
            1 if i in covered_indexes else 0 for i in range(len(input_tokens))
        ]
        return mask_labels

    def torch_mask_tokens(
            self,
            inputs: Any,
            special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
        import torch

        labels = inputs.clone()  # [bz, len]
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(
            labels.shape,
            self.mlm_probability)  # [bz, seq_len], all values is 0.15
        special_tokens_mask = [[
            1 if token in self.exclude_tokens else 0 for token in val
        ] for val in labels.tolist()]
        special_tokens_mask = torch.tensor(special_tokens_mask,
                                           dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        ### add by ruihan.wjn relace all values to 0 where belongs to kg_prompt

        ###
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer),
                                     labels.shape,
                                     dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class WikiKPPLMSupervisedJsonProcessor(DataProcessor):
    def __init__(self, data_args, training_args, model_args):
        super().__init__(data_args, training_args, model_args)

        # try:
        #     self.is_only_mlm = training_args.is_only_mlm
        # except:
        #     self.is_only_mlm = False
        self.is_only_mlm = False

    def get_data_collator(self):

        pad_to_multiple_of_8 = self.data_args.line_by_line and self.training_args.fp16 and not self.data_args.pad_to_max_length

        # return DataCollatorForWikiKGPLM(
        #     tokenizer=self.tokenizer,
        #     mlm_probability=self.data_args.mlm_probability,
        #     pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        # )
        if self.is_only_mlm:
            print('You set is_only_mlm is True')
            return DataCollatorForProcessedWikiKGPLM_OnlyMLM(
                tokenizer=self.tokenizer,
                mlm_probability=self.data_args.mlm_probability,
                pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
            )
        print('You set is_only_mlm is False')
        return DataCollatorForProcessedWikiKGPLM(
            tokenizer=self.tokenizer,
            mlm_probability=self.data_args.mlm_probability,
            pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        )

    def get_examples(self, set_type=None):
        data_files = {}
        if self.data_args.train_file is not None:
            data_files['train'] = self.data_args.train_file
            extension = self.data_args.train_file.split('.')[-1]
        if self.data_args.validation_file is not None:
            data_files['validation'] = self.data_args.validation_file
            extension = self.data_args.validation_file.split('.')[-1]
        if extension == 'json':
            extension = 'json'
        raw_datasets = load_dataset(extension,
                                    data_files=data_files,
                                    cache_dir=self.model_args.cache_dir)
        # raw_datasets['train'] = raw_datasets['train'].shuffle()
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if 'validation' not in raw_datasets.keys():
            raw_datasets['validation'] = load_dataset(
                extension,
                data_files=data_files,
                split=f'train[:{self.data_args.validation_split_percentage}%]',
                cache_dir=self.model_args.cache_dir,
            )
            raw_datasets['train'] = load_dataset(
                extension,
                data_files=data_files,
                split=f'train[{self.data_args.validation_split_percentage}%:]',
                cache_dir=self.model_args.cache_dir,
            )
        return raw_datasets

    def compute_metrics(self, p: EvalPrediction):
        # print('p.label_ids=', p.label_ids)
        # print('type(p.label_ids)=', type(p.label_ids))
        # print('p.predictions=', p.predictions)
        # print('type(p.predictions)=', type(p.predictions))
        if type(p.predictions) in [tuple, list]:
            preds = p.predictions[1]
        else:
            preds = p.predictions
        preds = preds[p.label_ids != -100]
        labels = p.label_ids[p.label_ids != -100]
        acc = (preds == labels).mean()
        return {'acc': round(acc, 4)}

    def get_tokenized_datasets(self):

        data_files = {}
        if self.data_args.train_file is not None:
            data_files['train'] = self.data_args.train_file
            extension = self.data_args.train_file.split('.')[-1]
        if self.data_args.validation_file is not None:
            data_files['validation'] = self.data_args.validation_file
            extension = self.data_args.validation_file.split('.')[-1]
        if extension == 'json':
            extension = 'json'
        raw_datasets = load_dataset(extension,
                                    data_files=data_files,
                                    cache_dir=self.model_args.cache_dir)
        # raw_datasets['train'] = raw_datasets['train'].shuffle()
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if 'validation' not in raw_datasets.keys():
            raw_datasets['validation'] = load_dataset(
                extension,
                data_files=data_files,
                split=f'train[:{self.data_args.validation_split_percentage}%]',
                cache_dir=self.model_args.cache_dir,
            )
            raw_datasets['train'] = load_dataset(
                extension,
                data_files=data_files,
                split=f'train[{self.data_args.validation_split_percentage}%:]',
                cache_dir=self.model_args.cache_dir,
            )
        logger.info(f'validation fingerprint {raw_datasets}')
        '''
        e.g.
        raw_datasets = DatasetDict({
            train: Dataset({
                features: ['json'],
                num_rows: xxx
            })
            validation: Dataset({
                features: ['json'],
                num_rows: xxx
            })
        })
        '''

        if self.training_args.do_train:
            column_names = raw_datasets['train'].column_names
        else:
            column_names = raw_datasets['validation'].column_names
        text_column_name = 'text' if 'text' in column_names else column_names[0]
        max_seq_length = self.tokenizer.model_max_length if self.data_args.max_seq_length is None else self.data_args.max_seq_length
        # When using line_by_line, we just tokenize each nonempty line.
        padding = 'max_length' if self.data_args.pad_to_max_length else False

        tokenizer = self.tokenizer

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name]
                if len(line) > 0 and not line.isspace()
            ]
            # examples['length'] = [len(line) for line in examples[text_column_name]]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )

        # with self.training_args.main_process_first(desc="dataset map tokenization"):
        #     tokenized_datasets = raw_datasets.map(
        #         tokenize_function,
        #         batched=True,
        #         num_proc=self.data_args.preprocessing_num_workers,
        #         remove_columns=[text_column_name],
        #         load_from_cache_file=not self.data_args.overwrite_cache,
        #         desc="Running tokenizer on dataset line_by_line",
        #     )
        '''
        {
            'text': tokens_str,
            'entity_ids': entity_ids,
            'mention_spans': mention_spans
        }
        '''
        return raw_datasets
