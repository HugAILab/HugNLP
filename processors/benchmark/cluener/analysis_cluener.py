# 本文件用于分析CLUE NER的数据分布情况

import os
import numpy as np
from tqdm import tqdm
import json

path = '../../datasets/cluener_public'
train_file = os.path.join(path, 'train.json')
dev_file = os.path.join(path, 'dev.json')
dev_predict_file = os.path.join(path, 'cluener_dev_predict.json')


def read_json(input_file):
    """Reads a json list file."""
    with open(input_file, 'r') as f:
        reader = f.readlines()
        lines = []
        for line in reader:
            lines.append(json.loads(line.strip()))
        return lines


def process(lines):
    '''
        train/dev:
        {
            "text": "而且还计划将民生银行作为战略投资者引入进来。",
            "label": {"company": {"民生银行": [[6, 9]]}, "position": {"战略投资者": [[12, 16]]}}
        }
        test:
        {
            "id": 1,
            "text": "而且还计划将民生银行作为战略投资者引入进来。"
        }
    '''
    entity2type = dict()
    for line in lines:
        text: str = line['text']  # 原始文本
        target: dict = line['label'] if 'label' in line.keys() else dict()
        for type, ent_spans in target.items():
            for entity, span in ent_spans.items():
                if entity not in entity2type.keys():
                    entity2type[entity] = dict()
                if type not in entity2type[entity].keys():
                    entity2type[entity][type] = 0
                entity2type[entity][type] += 1
    return entity2type


def pred_analysis(pred_json, dev_json):
    type_loss = 0  # 预测结果漏掉某一类实体
    type_add = 0  # 预测结果多处某一类
    ent_loss = 0
    ent_add = 0
    span_loss = 0
    span_add = 0
    for preds, dev in zip(pred_json, dev_json):
        pred: dict = preds['label']
        target: dict = dev['label']
        for type, ent_spans in target.items():
            if type not in pred.keys():
                type_loss += 1
                continue
            pred_ent_spans: dict = pred[type]
            for ent, spans in ent_spans.items():
                if ent not in pred_ent_spans.keys():
                    ent_loss += 1
                    continue
                pred_spans = pred_ent_spans[ent]
                for span in spans:
                    if span not in pred_spans:
                        span_loss += 1
    print('type_loss={}'.format(type_loss))
    print('type_add={}'.format(type_add))
    print('ent_loss={}'.format(ent_loss))
    print('ent_add={}'.format(ent_add))
    print('span_loss={}'.format(span_loss))
    print('span_add={}'.format(span_add))


def pred_analysis2(pred_json, dev_json):
    '''
            train/dev:
            {
                "text": "而且还计划将民生银行作为战略投资者引入进来。",
                "label": {"company": {"民生银行": [[6, 9]]}, "position": {"战略投资者": [[12, 16]]}}
            }
        '''
    ent_all = 0
    ent_type_error = 0  # 实体被预测出来，但是类别错误
    ent_span_error = 0  # 实体被预测出来，但是span错误
    ent_loss = 0  # 实体预测结果遗漏的数量
    ent_add = 0  # 预测出来的实体不在正确标签里，即预测错误的
    for preds, dev in zip(pred_json, dev_json):
        pred: dict = preds['label']
        target: dict = dev['label']
        # print(pred)
        # print(target)
        # print("*" * 20)
        entity2type = dict()
        entity2span = dict()
        for type, ent_spans in target.items():
            for ent, spans in ent_spans.items():
                entity2type[ent] = type
                entity2span[ent] = spans
        # 检查预测的结果与标签差异
        for type, ent_spans in pred.items():
            for ent, spans in ent_spans.items():
                ent_all += 1
                if ent not in entity2type:
                    print('ent=', ent)
                    print('entity2type=', entity2type)
                    print('*' * 20)
                    ent_add += 1
                    continue
                if type != entity2type[ent]:
                    ent_type_error += 1
                span_yes = True
                for span in spans:
                    if span not in entity2span[ent]:
                        span_yes = False
                if not span_yes:
                    ent_span_error += 1
    print('ent_type_error={}'.format(ent_type_error))
    print('ent_span_error={}'.format(ent_span_error))
    print('ent_loss={}'.format(ent_loss))
    print('ent_add={}'.format(ent_add))
    print('ent_all={}'.format(ent_all))


if __name__ == '__main__':
    # 分析训练集和验证集的标注情况
    train_json = read_json(train_file)
    dev_json = read_json(dev_file)
    entity2type = process(train_json)
    num = 0
    # for entity, type_dict in entity2type.items():
    #     if len(type_dict.keys()) > 1:
    #         num += 1
    #         print("*" * 20)
    #         print("entity={}".format(entity))
    #         print("type dict={}".format(type_dict))
    # print("multi-label num={}".format(num))

    # 分析模型预测的验证集结果与验证集的差异。（错误分析）
    dev_predict_json = read_json(dev_predict_file)
    pred_analysis(dev_predict_json, dev_json)
    print('======')
    pred_analysis2(dev_predict_json, dev_json)
