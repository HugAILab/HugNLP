import sys
import os
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
import json
from collections import defaultdict, Counter
# from cpic_process import CPICProcessor

base_dir = "/wjn/competition/clue/nlp_runner/outputs/finetune_b"
# base_dir = "/wjn/competition/clue/nlp_runner/outputs/finetune"
predict_result_files = [
    "chinese-macbert-large", "chinese_pretrain_mrc_macbert_large",
    "chinese-roberta-wwm-ext-large", "chinese_pretrain_mrc_roberta_wwm_ext_large",
    "chinese-pert-large", "chinese-pert-large-mrc", "structbert-large-zh"
]

# 根据所有预测结果的概率进行模型融合
def merge_output(predict_result_files):
    out = defaultdict(Counter)
    pos = dict()
    for result_file in predict_result_files:
        path = os.path.join(base_dir, result_file, "topk_prob.json")
        result = json.load(open(path, encoding="utf8"))
        for key, value in result.items():
            if key not in pos.keys():
                pos[key] = dict()
            for v in value:
                out[key][v["answer"]] += v["prob"]
                pos[key][v["answer"]] = v["pos"] # 对应的样本的答案的start/end位置（元组）
    best = {k: v.most_common()[0][0] for k, v in out.items() if "ner" not in k.lower()}
    for k, v in out.items():
        if "ner" in k.lower():
            best[k] = [m for m, n in v.items() if n > 0.6 * len(predict_result_files)]
        else:
            best[k] = v.most_common()[0][0]
    return out, best, pos

# 根据预测的结果，根据阈值筛划分数据（高置信度的覆盖率越小，准确率会越大）
def obtain_confused_result(out: defaultdict, threshole=0.6, confuse_value=0.15, num_template=2):
    # num_template: 需要调整这个值，测试集的每个example对应设计的template个数
    # 遍历每一个样本，
    # 1。如果存在预测的部分答案概率很接近，则说明该答案很难预测
    # 2。如果概率最大的预测结果概率值低于阈值，则认为该样本很难预测
    hard_answer = dict()
    for k, v in out.items():
        if "ner" in k.lower():
            continue
        is_filter = False
        best_result, best_prob = v.most_common()[0][0], v.most_common()[0][1]
        second_result, second_prob = v.most_common()[1][0], v.most_common()[1][1]
        if best_prob <= threshole * len(predict_result_files) * num_template:
            is_filter = True
        if abs(best_prob - second_prob) < confuse_value * len(predict_result_files) * num_template:
            is_filter = True
        if is_filter:
            num = len(v.most_common())
            hard_answer[k] = {v.most_common()[i][0]: v.most_common()[i][1] for i in range(num)}
    print("hard answer num: {}".format(len(hard_answer.keys())))
    return hard_answer



if __name__ == "__main__":
    # save_path = "/wjn/competition/clue/nlp_runner/outputs/finetune/merge_answer_result_two_template5"
    # save_path = "/wjn/competition/clue/nlp_runner/outputs/finetune/merge_answer_result5"
    save_path = "/wjn/competition/clue/nlp_runner/outputs/finetune_b/merge_answer_result3"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    outfile = os.path.join(save_path, "answer.json")
    outfile2 = os.path.join(save_path, "answer_prob.json")
    outfile3 = os.path.join(save_path, "hard_answer.json")
    outfile4 = os.path.join(save_path, "answer_pos.json")
    out, best, pos = merge_output(predict_result_files)
    hard_answer = obtain_confused_result(out)
    with open(outfile, "w", encoding="utf8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    with open(outfile2, "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(outfile3, "w", encoding="utf8") as f:
        json.dump(hard_answer, f, ensure_ascii=False, indent=2)
    with open(outfile4, "w", encoding="utf8") as f:
        json.dump(pos, f, ensure_ascii=False, indent=2)
