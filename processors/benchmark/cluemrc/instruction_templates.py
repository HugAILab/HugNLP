import json
import sys
import os
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
from processors.cluemrc.instruction_prompts import *
from processors.cluemrc.clue_processor import get_all_data, clue_path
from tqdm import tqdm

clue_instruction = {
    'afqmc': {
        'prompt': '以下两句话的内容是否相似？{}。【{}】和【{}】',
        'keys_order': ['verbalizer', 'text_a', 'text_b'],
        'instruction': PARAInstruction,
        'data_type': 'classification',
    },
    'cmnli': {
        'prompt': '以下两句话的逻辑关系是什么？{}。【{}】和【{}】',
        'keys_order': ['verbalizer', 'text_a', 'text_b'],
        'instruction': NLIInstruction,
        'data_type': 'classification',
    },
    'ocnli': {
        'prompt': '以下两句话的逻辑关系是什么？{}。【{}】和【{}】',
        'keys_order': ['verbalizer', 'text_a', 'text_b'],
        'instruction': NLIInstruction,
        'data_type': 'classification',
    },
    'csl': {
        'prompt': '文章可以用【{}】作为关键词吗？{}。文章:【{}】',
        'keys_order': ['text_a', 'verbalizer', 'text_b'],
        'instruction': ClassificationInstruction,
        'data_type': 'classification',
    },
    'iflytek': {
        'prompt': '文章的类别是什么？{}。【{}】',
        'keys_order': ['verbalizer', 'text_a'],
        'instruction': ClassificationInstruction,
        'data_type': 'classification',
    },
    'tnews': {
        'prompt': '文章的类别是什么？{}。【{}】',
        'keys_order': ['verbalizer', 'text_a'],
        'instruction': ClassificationInstruction,
        'data_type': 'classification',
    },
    'wsc': {
        'prompt': '文章中【{}】的是【{}】吗？{}。文章:【{}】',
        'keys_order': ['span2_text', 'span1_text', 'verbalizer', 'text_a'],
        'instruction': WSCInstruction,
        'data_type': 'classification',
    },
    'cmrc': {
        'prompt': '找到问题【{}】的答案？文章:【{}】',
        'keys_order': ['question', 'context'],
        'instruction': MRCInstruction,
        'data_type': 'mrc',
    },
    'c3': {
        'prompt': '阅读文章回答问题【{}】应选择:{}。文章:【{}】',
        'keys_order': ['question', 'choice', 'context'],
        'instruction': MultiChoiceInstruction,
        'data_type': 'classification',
    },
    'bq_corpus': {
        'prompt': '以下两句话的内容是否相似？{}。【{}】和【{}】',
        'keys_order': ['verbalizer', 'text_a', 'text_b'],
        'instruction': PARAInstruction,
        'data_type': 'classification',
    },
    'chip-cdn': {
        'prompt': '以下两句话的内容是否相似？{}。【{}】和【{}】',
        'keys_order': ['verbalizer', 'text_a', 'text_b'],
        'instruction': PARAInstruction,
        'data_type': 'classification',
    },
    'chip-sts': {
        'prompt': '以下两句话的内容是否相似？{}。【{}】和【{}】',
        'keys_order': ['verbalizer', 'text_a', 'text_b'],
        'instruction': PARAInstruction,
        'data_type': 'classification',
    },
    'covid_emotion': {
        'prompt': '文章的情感态度是什么？{}。【{}】',
        'keys_order': ['verbalizer', 'text_a'],
        'instruction': ClassificationInstruction,
        'data_type': 'classification',
    },
    'lcqmc': {
        'prompt': '以下两句话的内容是否相似？{}。【{}】和【{}】',
        'keys_order': ['verbalizer', 'text_a', 'text_b'],
        'instruction': PARAInstruction,
        'data_type': 'classification',
    },
    'oppo': {
        'prompt': '以下两句话的内容是否相似？{}。【{}】和【{}】',
        'keys_order': ['verbalizer', 'text_a', 'text_b'],
        'instruction': PARAInstruction,
        'data_type': 'classification',
    },
    'squad': {
        'prompt': '找到问题【{}】的答案？文章:【{}】',
        'keys_order': ['question', 'context'],
        'instruction': MRCInstruction,
        'data_type': 'mrc',
    },
    'car_emotion': {
        'prompt': '文章的情感态度是什么？{}。【{}】',
        'keys_order': ['verbalizer', 'text_a'],
        'instruction': ClassificationInstruction,
        'data_type': 'classification',
    },
    'chip-ctc': {
        'prompt': '文章的类别是什么？{}。【{}】',
        'keys_order': ['verbalizer', 'text_a'],
        'instruction': ClassificationInstruction,
        'data_type': 'classification',
    },
    'cmeee': {
        'prompt': '找到文章中所有【{}】类型的实体？文章:【{}】',
        'keys_order': ['entity_type', 'context'],
        'instruction': NERInstruction,
        'data_type': 'ner',
    },
    'kuake-qic': {
        'prompt': '文章的类别是什么？{}。【{}】',
        'keys_order': ['verbalizer', 'text_a'],
        'instruction': ClassificationInstruction,
        'data_type': 'classification',
    },
    'paws-x-zh': {
        'prompt': '以下两句话的内容是否相似？{}。【{}】和【{}】',
        'keys_order': ['verbalizer', 'text_a', 'text_b'],
        'instruction': PARAInstruction,
        'data_type': 'classification',
    },
    'turing_emotion': {
        'prompt': '文章的情感态度是什么？{}。【{}】',
        'keys_order': ['verbalizer', 'text_a'],
        'instruction': ClassificationInstruction,
        'data_type': 'classification',
    },
}

other_instruction = {
    'senti': {
        'prompt': '文章的情感态度是什么？{}。【{}】',
        'keys_order': ['verbalizer', 'text_a'],
        'instruction': ClassificationInstruction,
        'data_type': 'classification',
    },
    'cls': {
        'prompt': '文章的类别是什么？{}。【{}】',
        'keys_order': ['verbalizer', 'text_a'],
        'instruction': ClassificationInstruction,
        'data_type': 'classification',
    },
    'app': {
        'prompt': '文章的类别是什么？{}。【{}】',
        'keys_order': ['verbalizer', 'text_a'],
        'instruction': ClassificationInstruction,
        'data_type': 'classification',
    },
    'news': {
        'prompt': '文章的类别是什么？{}。【{}】',
        'keys_order': ['verbalizer', 'text_a'],
        'instruction': ClassificationInstruction,
        'data_type': 'classification',
    },
    'intent': {
        'prompt': '这句话的意图是什么？{}。【{}】',
        'keys_order': ['verbalizer', 'text_a'],
        'instruction': ClassificationInstruction,
        'data_type': 'classification',
    },
    'nli': {
        'prompt': '以下两句话的逻辑关系是什么？{}。【{}】和【{}】',
        'keys_order': ['verbalizer', 'text_a', 'text_b'],
        'instruction': NLIInstruction,
        'data_type': 'classification',
    },
    'sts': {
        'prompt': '以下两句话的内容是否相似？{}。【{}】和【{}】',
        'keys_order': ['verbalizer', 'text_a', 'text_b'],
        'instruction': STSInstruction,
        'data_type': 'classification',
    },
    'para': {
        'prompt': '以下两句话的内容是否相似？{}。【{}】和【{}】',
        'keys_order': ['verbalizer', 'text_a', 'text_b'],
        'instruction': PARAInstruction,
        'data_type': 'classification',
    },
    'mrc': {
        'prompt': '找到问题【{}】的答案？文章:【{}】',
        'keys_order': ['question', 'context'],
        'instruction': MRCInstruction,
        'data_type': 'mrc',
    },
    'ner': {
        'prompt': '找到文章中所有【{}】类型的实体？文章:【{}】',
        'keys_order': ['entity_type', 'context'],
        'instruction': NERInstruction,
        'data_type': 'ner',
    },
    'summ': {
        'prompt': '文章的摘要是什么？文章:【{}】',
        'keys_order': ['passage'],
        'instruction': SUMMInstruction,
        'data_type': 'summ',
    },
    'keys': {
        'prompt': '找到文章的关键词？文章:【{}】',
        'keys_order': ['text_a'],
        'instruction': KEYSInstruction,
        'data_type': 'keys',
    },
    'wsc': {
        'prompt':
        '文章中【{}】的是【{}】吗？{}。文章:【{}】',
        'keys_order':
        ['target/span2_text', 'target/span1_text', 'verbalizer', 'text'],
        'instruction':
        WSCInstruction,
        'data_type':
        'classification',
    },
    'yesno': {
        'prompt': '阅读文章回答问题【{}】？应选择:{}。文章:【{}】',
        'keys_order': ['text_a', 'choice', 'text_b'],
        'instruction': MultiChoiceInstruction,
        'data_type': 'classification',
    },
    'c3': {
        'prompt': '阅读文章回答问题【{}】应选择:{}。文章:【{}】',
        'keys_order': ['question', 'choice', 'context'],
        'instruction': MultiChoiceInstruction,
        'data_type': 'classification',
    },
    'weibo_emotion': {
        'prompt': '文章的情感态度是什么？{}。文章:【{}】',
        'keys_order': ['verbalizer', 'text_a'],
        'instruction': WeiboEmotionInstruction,
        'data_type': 'classification',
    },
    'lsht': {
        'prompt': '文章的类别是什么？{}。文章:【{}】',
        'keys_order': ['verbalizer', 'content'],
        'instruction': ClassificationInstruction,
        'data_type': 'classification',
    }
}


def instruction_format(data_dict: Dict) -> List[Dict]:
    special_datasets = {
        'dureader_yesno': 'yesno',
        'c3': 'c3',
        'cail_yesno': 'c3',
        'NLPCC2014_Weibo_Emotion_classification': 'weibo_emotion',
        'NLPCC2014_LSHT_sample': 'lsht'
    }
    instruction_data = []
    for data_type, type_dict in tqdm(data_dict.items()):
        # 关键词抽取任务keys很蠢，不要了。summ属于生成，不适用于抽取
        if data_type in ['keys', 'summ']:
            continue
        for data_name, data_info in type_dict.items():
            # 纯英文的不要
            if data_name in ['intent_classification']:
                continue

            label_mappings = data_info.get('label_mappings')
            # print('label_mappings=', label_mappings)
            label_mappings = {
                key: value[0] if type(value) == list else value
                for key, value in label_mappings.items()
            }
            data_list = data_info['data_list']
            for instruction in [clue_instruction]:
                format_info = instruction[special_datasets.get(
                    data_name, data_type)]
                instruction_processor = format_info['instruction'](
                    data_name, data_list, label_mappings,
                    format_info['prompt'], format_info['keys_order'],
                    format_info['data_type'])
                instruction_data.extend(
                    instruction_processor.transform2instruction())
                # print(instruction_data[-1])

    return instruction_data


if __name__ == '__main__':

    base_path = '/wjn/competition/clue/datasets'

    # CLUE榜单的数据集读取，并转换为instruction形式，最后转换为mrc格式
    all_training_data, clue_train_data, clue_dev_data, clue_test_data = get_all_data(
        base_path)
    # add merge datai
    print('merge data')
    merge_data = instruction_format(all_training_data)
    if not os.path.exists(os.path.join(base_path, 'merge')):
        os.makedirs(os.path.join(base_path, 'merge'))
    with open(os.path.join(base_path, 'merge', 'clue_train.json'),
              'w',
              encoding='utf8') as fout:
        json.dump(merge_data, fout, indent=2, ensure_ascii=False)
    # add single task
    clue_tasks = list(clue_train_data.keys())
    for clue_task in clue_tasks:
        print('single data: {}'.format(clue_task))
        train_data = instruction_format(clue_train_data[clue_task])
        dev_data = instruction_format(clue_dev_data[clue_task])
        test_data = instruction_format(clue_test_data[clue_task])
        if not os.path.exists(
                os.path.join(base_path, clue_path[clue_task], 'mrc_style/')):
            os.makedirs(
                os.path.join(base_path, clue_path[clue_task], 'mrc_style/'))
        with open(os.path.join(base_path, clue_path[clue_task], 'mrc_style/',
                               'train.json'),
                  'w',
                  encoding='utf8') as fout:
            json.dump(train_data, fout, indent=2, ensure_ascii=False)
        with open(os.path.join(base_path, clue_path[clue_task], 'mrc_style/',
                               'dev.json'),
                  'w',
                  encoding='utf8') as fout:
            json.dump(dev_data, fout, indent=2, ensure_ascii=False)
        with open(os.path.join(base_path, clue_path[clue_task], 'mrc_style/',
                               'test.json'),
                  'w',
                  encoding='utf8') as fout:
            json.dump(test_data, fout, indent=2, ensure_ascii=False)

    # 其他other数据集，已经转换为instruction的形式了，直接转为mrc格式即可
    # print("other data:")
    # other_data = list()
    # for ori_file in ['bq_corpus.json', 'chip-cdn.json', 'chip-sts.json', 'covid_emotion.json',
    #                    'lcqmc.json', 'paws-x-zh.json', 'turing_emotion.json', 'car_emotion.json',
    #                    'chip-ctc.json', 'cmeee.json', 'kuake-qic.json', 'oppo.json', 'squad.json']:
    #     print("others: single data: {}".format(ori_file))
    #     data = json.load(open(os.path.join(base_path, "others", ori_file), encoding='utf8'))
    #     add_type_data = {
    #         "{}".format(ori_file.split(".")[0]): {
    #             "{}".format(ori_file.split(".")[0]) :data
    #         }
    #     }
    #     instruction_data = instruction_format(add_type_data)
    #     other_data.extend(instruction_data)
    # with open(os.path.join(base_path, "merge", "others_train.json"), 'w', encoding='utf8') as fout:
    #     json.dump(other_data, fout, indent=2, ensure_ascii=False)
