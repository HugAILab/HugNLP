import numpy as np


def calculate_metric(class_res, type='', threshold=0.1):
    # 如果detection预测其为实体，但是proto给出的预测概率低于90%，则直接预测其为'O'

    # 计算token级别NER识别结果
    detection_pred_cnt = 0  # detection预测为positive的个数
    detection_label_cnt = 0  # label总数
    detection_true_cnt = 0  # label为positive的个数
    detection_acc = 0
    detection_precision = 0
    detection_recall = 0

    for token_i in range(len(class_res)):
        pred_j, label_j = class_res[token_i][0], class_res[token_i][1]
        if len(class_res[token_i]) == 3:
            pred_proba_j = class_res[token_i][2]
            pred_j = 0 if pred_proba_j <= threshold else pred_j
        if label_j == -1:
            continue
        else:
            detection_label_cnt += 1
            if int(label_j) > 0:
                # print('label_j=', label_j)
                # print('pred_j=', pred_j)
                detection_true_cnt += 1
                if pred_j == label_j:
                    detection_recall += 1
            if int(pred_j) > 0:
                # print('label_j=', label_j)
                # print('pred_j=', pred_j)
                detection_pred_cnt += 1
                if pred_j == label_j:
                    detection_precision += 1
            if pred_j == label_j:
                detection_acc += 1

    print('detection_pred_cnt=', detection_pred_cnt)
    print('detection_precision=', detection_precision)
    print('detection_true_cnt=', detection_true_cnt)
    print('detection_recall=', detection_recall)

    detection_acc = round(detection_acc / detection_label_cnt, 4)
    detection_precision = round(detection_precision / detection_pred_cnt, 4)
    detection_recall = round(detection_recall / detection_true_cnt, 4)
    detection_f1 = 2 * detection_precision * detection_recall / (
        detection_precision + detection_recall)
    detection_f1 = round(detection_f1, 4)

    print('[EVAL-{} | acc: {}, precision: {}, recall: {}, f1: {}'.format(
        type, detection_acc, detection_precision, detection_recall,
        detection_f1))


if __name__ == '__main__':
    with open('output_result.txt', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()[1:-1]
    token_res = list()
    class_res = list()
    class_res2 = list()
    with open('output_errors.txt', 'w', encoding='utf-8') as fw:
        for i in lines:
            try:
                det_pred, det_prob, det_label, cls_pred, cls_logit, cls_label = i.replace(
                    '\n', '').split('\t')
                token_res.append(
                    [int(det_pred),
                     int(det_label),
                     float(det_prob)])
                class_res.append(
                    [int(cls_pred),
                     int(cls_label),
                     float(cls_logit)])
                class_res2.append([int(cls_pred), int(cls_label)])
            except:
                # print(i)
                continue
            # if cls_pred != cls_label and cls_label == 0:
            #     fw.write(i)

    calculate_metric(token_res, 'Detection', threshold=0.1)
    calculate_metric(class_res, 'CLassification', threshold=0.1)
    calculate_metric(class_res2, 'CLassification', threshold=0.1)
