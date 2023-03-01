import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../..'))
from random import shuffle
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 负责生成in-context format训练语料
import os
import json
from tqdm import tqdm
# from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from processors.knowledge_graph.wikidata5m.utils import Wikidata5m
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.model_max_length

params = {
    'anli': {
        'prompt_prefix':
        'Classify two sentences into the categories of entailment, neutral and contradiction.\n\n',
        'q_prefix': 'Texts: ',
        'a_prefix': 'Output: ',
    },
    'app_reviews': {
        'prompt_prefix': 'Classify the review.\n\n',
        'q_prefix': 'Review: ',
        'a_prefix': 'Sentiment: ',
        'label_word': {
            '1': 'awful',
            '2': 'better',
            '3': 'hilarious',
            '4': 'remarkable',
            '5': 'wonderful'
        },
    },
    'boolq': {
        'prompt_prefix': 'Answer the question with yes or no.\n\n',
        'q_prefix': '',
        'a_prefix': 'Answer: ',
    },
    'circa': {
        'prompt_prefix': '',
        'q_prefix': 'Is the answer correct?\n\n',
        'a_prefix': 'Output: ',
    },
    'codah': {
        'prompt_prefix': 'Answer the question through multiple-choice.\n\n',
        'q_prefix': 'Question: ',
        'a_prefix': 'Answer: ',
    },
    'common_gen': {
        'prompt_prefix': 'Generate text based on multiple entities.\n\n',
        'q_prefix': 'Entities: ',
        'a_prefix': 'Text: ',
    },
    'dbpedia_14': {
        'prompt_prefix': 'Classify the topic of each text.\n\n',
        'q_prefix': 'Text: ',
        'a_prefix': 'Topic: ',
    },
    'glue-wnli': {
        'prompt_prefix':
        'Classify two sentences into the categories of entailment, neutral and contradiction.\n\n',
        'q_prefix': 'Texts: ',
        'a_prefix': 'Output: ',
    },
    'imdb': {
        'prompt_prefix': 'Classify the review.\n\n',
        'q_prefix': 'Review: ',
        'a_prefix': 'Sentiment: ',
    },
    'kilt_hotpotqa': {
        'prompt_prefix': 'Answer the question.\n\n',
        'q_prefix': 'Question: ',
        'a_prefix': 'Answer: ',
    },
    'kilt_nq': {
        'prompt_prefix': 'Answer the question.\n\n',
        'q_prefix': 'Question: ',
        'a_prefix': 'Answer: ',
    },
    'math_qa': {
        'prompt_prefix': 'Choose the correct answer of math word problem.\n\n',
        'q_prefix': 'Question: ',
        'a_prefix': 'Answer: ',
    },
    'medical_questions_pairs': {
        'prompt_prefix': 'Whether the two question are similar?\n\n',
        'q_prefix': 'Questions: ',
        'a_prefix': 'Output: ',
    },
    'openbookqa': {
        'prompt_prefix': 'Answer the question through multiple-choice.\n\n',
        'q_prefix': 'Question: ',
        'a_prefix': 'Answer: ',
    },
    'paws': {
        'prompt_prefix': 'Whether the two question are similar?\n\n',
        'q_prefix': 'Questions: ',
        'a_prefix': 'Output: ',
        'label_word': {
            '0': 'No',
            '1': 'Yes'
        },
    },
    'piqa': {
        'prompt_prefix': 'Whether the two question are similar?\n\n',
        'q_prefix': 'Questions: ',
        'a_prefix': 'Output: ',
        'label_word': {
            '0': 'No',
            '1': 'Yes'
        },
    },
    'qasc': {
        'prompt_prefix': 'Answer the question through multiple-choice.\n\n',
        'q_prefix': 'Question: ',
        'a_prefix': 'Answer: ',
    },
    'quarel': {
        'prompt_prefix': 'Answer the question through multiple-choice.\n\n',
        'q_prefix': 'Question: ',
        'a_prefix': 'Answer: ',
    },
    'quartz-no_knowledge': {
        'prompt_prefix': 'Answer the question through multiple-choice.\n\n',
        'q_prefix': 'Question: ',
        'a_prefix': 'Answer: ',
    },
    'rotten_tomatoes': {
        'prompt_prefix': 'Classify the review.\n\n',
        'q_prefix': 'Review: ',
        'a_prefix': 'Sentiment: ',
    },
    'sciq': {
        'prompt_prefix': 'Answer the question through multiple-choice.\n\n',
        'q_prefix': 'Question: ',
        'a_prefix': 'Answer: ',
    },
    'scitail': {
        'prompt_prefix':
        'Classify two sentences into the categories of entailment, neutral and contradiction.\n\n',
        'q_prefix': 'Texts: ',
        'a_prefix': 'Output: ',
    },
    'sick': {
        'prompt_prefix':
        'Classify two sentences into the categories of entailment, neutral and contradiction.\n\n',
        'q_prefix': 'Texts: ',
        'a_prefix': 'Output: ',
    },
    'social_i_qa': {
        'prompt_prefix': 'Answer the question through multiple-choice.\n\n',
        'q_prefix': 'Question: ',
        'a_prefix': 'Answer: ',
    },
    'superglue-cb': {
        'prompt_prefix':
        'Classify two sentences into the categories of entailment, neutral and contradiction.\n\n',
        'q_prefix': 'Texts: ',
        'a_prefix': 'Output: ',
    },
    'superglue-copa': {
        'prompt_prefix': 'Answer the question through multiple-choice.\n\n',
        'q_prefix': 'Question: ',
        'a_prefix': 'Answer: ',
    },
    'superglue-rte': {
        'prompt_prefix':
        'Classify two sentences into the categories of entailment, neutral and contradiction.\n\n',
        'q_prefix': 'Texts: ',
        'a_prefix': 'Output: ',
        'label_word': {
            'not_entailment': 'contradiction',
            'entailment': 'entailment'
        }
    },
    'yelp_polarity': {
        'prompt_prefix': 'Classify the review.\n\n',
        'q_prefix': 'Review: ',
        'a_prefix': 'Sentiment: ',
    }
}


def construct_prompt_token(params,
                           tokenizer: GPT2TokenizerFast,
                           train_datasets: list,
                           only_train_last: bool = False,
                           max_len: int = 1000):
    # take the prompt template and fill in the training and test example
    '''
    e.g. train_datasets
    [
        {
            "task_name": "aeslc",
            "task_type": "qa",
            "task_data": [
                {
                    "idx": 0,
                    "text": "summarize: All, Please feel free to test enpower NETCO production and we will truncate deal data later. If you found problems or comments, please let us know asap. Windows 2000 Shortcut for enpower NETCO production EnPower Launchpad for NETCO production EnPower Report Launcher for Contract Settlements Group EnPower Report Launcher for Credit Group EnPower Report Launcher for Deal Clearing Group EnPower Report Launcher for Risk Group EnPower Report Launcher for Scheduling Group EnPower Launchpad File Menu EnPower Launchpad Tools Menu EnPower Launchpad Downloads Menu Regards,",
                    "target": "EnPower NETCO production - TESTING"
                },
                ...
            ]
        },
        ...
    ]

    only_train_last: 一个样本对应多个context，当only_train_last为True时，表示只训练最后一个context（最后一个context label mask为1，其余为0）
    '''

    if only_train_last:
        print('Only set the last in-context example for training.')

    newline = tokenizer.encode('\n', add_special_tokens=False)
    newlines = tokenizer.encode('\n\n', add_special_tokens=False)

    all_prompt_data = list()  # 最终的保存样本

    param_unified = {
        'prompt_prefix': 'Answer the question.\n\n',
        'q_prefix': '',
        'a_prefix': 'Answer: ',
    }

    qa_data_num, cls_data_num = 0, 0

    for task_dict in tqdm(train_datasets):
        # 遍历每一个task
        task_name = task_dict['task_name']
        task_type = task_dict['task_type']
        task_data = task_dict['task_data']
        shuffle(task_data)
        param = params[task_name] if task_name in params.keys(
        ) else param_unified

        prompt_prefix = tokenizer.encode(
            param['prompt_prefix'], add_special_tokens=False
        ) if 'prompt_prefix' in param.keys() else []
        a_prefix = tokenizer.encode(
            param['a_prefix'],
            add_special_tokens=False) if 'a_prefix' in param.keys() else []

        cur_prompt = []  # 当前句子对应的模板
        cur_label_mask = []  # 当前句子的label mask

        cur_prompt += prompt_prefix
        cur_label_mask.append([0] * len(prompt_prefix))

        # 遍历所有data
        for example in task_data:
            # cur_sent_prompt = []
            # cur_sent_label_mask = []

            text = example['text'].replace('[SEP]', '\t')
            target = example['target']
            if 'label_word' in param.keys():
                label_word = param['label_word']
                target = label_word[target]
            text = param['q_prefix'] + text
            text_token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(text_token_ids) > max_len:  # text超限的进行截断
                text_token_ids = text_token_ids[:max_len - 100]
            target_token_ids = tokenizer.encode(target,
                                                add_special_tokens=False)
            # 获得当前样本的token id和mask id
            cur_sent_prompt = text_token_ids + newline + a_prefix + target_token_ids
            cur_sent_label_mask = [0] * (len(text_token_ids) + len(newline) +
                                         len(a_prefix))
            cur_sent_label_mask += [1] * len(target_token_ids)

            if len(cur_sent_prompt) > max_len:  # 总长度超限，则阶段生成的答案
                cur_sent_prompt = cur_sent_prompt[:max_len - 20]
                cur_sent_label_mask = cur_sent_label_mask[:max_len - 20]

            cur_sent_prompt += newlines
            cur_sent_label_mask += [0] * len(newlines)
            assert len(cur_sent_prompt) == len(cur_sent_label_mask)

            if (len(cur_prompt) + len(cur_sent_prompt)) > max_len:

                final_label_mask = list()
                if only_train_last and len(
                        cur_label_mask
                ) > 1:  # 如果只允许最后一个in-context example计算loss，则前面的所有example的label mask全部置0
                    for lm in cur_label_mask[:-1]:
                        final_label_mask.extend([0] * len(lm))
                elif not only_train_last and len(cur_label_mask) > 1:
                    for lm in cur_label_mask[:-1]:
                        final_label_mask.extend(lm)
                final_label_mask.extend(
                    cur_label_mask[-1])  # 最后一个in-context example一定需要计算loss

                all_prompt_data.append({
                    'token_ids': cur_prompt,
                    'label_masks': final_label_mask
                })

                if task_type == 'qa':
                    qa_data_num += 1
                else:
                    cls_data_num += 1

                cur_prompt = []  # 当前句子对应的模板
                cur_label_mask = []  # 当前句子的label mask
                cur_prompt += prompt_prefix
                cur_label_mask.append([0] * len(prompt_prefix))

            cur_prompt.extend(cur_sent_prompt)
            # label_mask.extend(cur_label_mask)
            cur_label_mask.append(cur_sent_label_mask)
    print('cls data num: {}'.format(cls_data_num))
    print('qa data num: {}'.format(qa_data_num))

    return all_prompt_data


if __name__ == '__main__':
    """本脚本用于生成question answer-style feature数据 python3 ./processor/pretraining/kg_enhance_causal_lm/generate_qa_data.py."""
    # === 数据生成控制
    only_train_last = False
    save_dir = '/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_nlp_task_data'
    if only_train_last is True:
        save_dir = '{}_only_train_last'.format(save_dir)
    print('save_dir=', save_dir)

    with open(
            '/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_nlp_task_data/all_data.json',
            'r',
            encoding='utf-8') as fr:
        data = json.load(fr)

    print('finish loading.')
    # shuffle(data)
    # print("finish shuffle.")

    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained('gpt2')

    all_prompt_data = construct_prompt_token(params,
                                             tokenizer,
                                             data,
                                             only_train_last=only_train_last)
    print('len(all_prompt_data)=', len(all_prompt_data))
    # shuffle(all_prompt_data)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'prompt_data.json'),
              'w',
              encoding='utf-8') as fw:
        for line in tqdm(all_prompt_data):
            fw.write('{}\n'.format(json.dumps(line, ensure_ascii=False)))
    # with open("./total_pretrain_data_gpt/prompt_data.json", 'w', encoding='utf8') as fw:
    #     json.dump(all_prompt_data, fw)
    '''
    e.g.
    {"token_ids": [14539, 871, 25, 220, 6929, 952, 509, 505, 87, 13, 198, 8206, 25, 220, 1375, 373, 3706, 366, 39, 34655, 22825, 310, ],
    "label_masks": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    '''
