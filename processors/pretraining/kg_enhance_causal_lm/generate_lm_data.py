import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../.."))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../.."))
from random import shuffle
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 负责生成in-context format训练语料
import os
import json
from tqdm import tqdm
# from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from processors.knowledge_graph.wikidata5m.utils import Wikidata5m

def construct_prompt_token(
    params,
    tokenizer: GPT2TokenizerFast,
    train_datasets: list,
    kg: Wikidata5m = None,
    only_train_last: bool = False):
    # take the prompt template and fill in the training and test example
    """
    train_datasets = [{
        "token_ids": [35082, 24041, 215, 25, 3303, 625, 853, 8, 26117, 1438, 260, 33, 4492, 8992, 13, 5, 10546,
        73, 18178, 5971, 1546, 4],
        "entity_pos": [[4, 7], [8, 11]],
        "entity_qid": ["Q170525", "Q185157"],
        "relation_pos": null,
        "relation_pid": null
    }, ...]

    only_train_last: 一个样本对应多个context，当only_train_last为True时，表示只训练最后一个context（最后一个context label mask为1，其余为0）
    """
    # assert type(tokenizer) == GPT2TokenizerFast
    if only_train_last:
        print("Only set the last in-context example for training.")
    max_len = params["max_len"] if "max_len" in params.keys() else 1016
    q_prefix = tokenizer.encode(params["q_prefix"], add_special_tokens=False) if "q_prefix" in params.keys() else []
    a_prefix = tokenizer.encode(params["a_prefix"], add_special_tokens=False) if "a_prefix" in params.keys() else []
    task_type = params["task_type"]

    newline = tokenizer.encode("\n", add_special_tokens=False)
    newlines = tokenizer.encode("\n\n", add_special_tokens=False)
    douhao = tokenizer.encode(",", add_special_tokens=False)
    juhao = tokenizer.encode(".", add_special_tokens=False)

    trigger_x_ids_str, trigger_y_ids_str = "", ""
    relation_pid2template_ids_str = dict()
    if kg is not None:
        trigger_x_ids_str, trigger_y_ids_str, relation_pid2template_ids_str = kg.tokenize_template()



    prompt = []
    label_mask = []
    all_prompt_data = list()

    for line in tqdm(train_datasets):
        token_ids = line["token_ids"]
        entity_pos = line["entity_pos"] # list
        entity_qids = line["entity_qid"] # list
        cur_prompt = [] # 当前句子对应的模板
        cur_label_mask = [] # 当前句子的label mask
        l_str = [] # 当前句子对应的标签
        cur_prompt += q_prefix
        cur_label_mask += [0] * len(q_prefix)
        if task_type == "mep": # Masked Entity Prediction (MEP)
            cur_prompt += token_ids + newline # 198表示\n
            l_str = [] # mask任务，没有标签
            a_prefix = []
            # 找到所有实体的位置，并设置为1
            cur_sent_label_mask = [0] * len(token_ids + newline)
            for s, e in entity_pos:
                for i in range(s, e):
                    cur_sent_label_mask[i] = 1
            cur_label_mask += cur_sent_label_mask
        elif task_type == "ecg": # Entity Conditional Generation（ECG）
            entity_tokens = list()
            for s, e in entity_pos:
                entity_tokens.extend(token_ids[s: e] + douhao) # 11表示逗号
            if len(entity_tokens) == 0:
                continue
            entity_tokens[-1] = juhao[0] # 最末尾的改为句号
            cur_prompt += entity_tokens + newline
            l_str = token_ids # 生成任务，标签是原始的句子
            cur_label_mask += [0] * len(entity_tokens + newline)
        elif task_type == "RKP": # Relational Knowledge Probing
            # 依次检索相邻的两个实体在KG中的三元组，如果不存在则剔除该句子
            if len(entity_pos) < 2: # 必须有至少两个实体
                continue
            for ent_i in range(len(entity_pos) - 1):
                ent_j = ent_i + 1
                ent_i_pos, ent_j_pos = entity_pos[ent_i], entity_pos[ent_j]
                entity_i_qid, entity_j_qid = entity_qids[ent_i], entity_qids[ent_j]
                entity_i_tokens = token_ids[ent_i_pos[0]: ent_i_pos[1]] # token ids
                entity_j_tokens = token_ids[ent_j_pos[0]: ent_j_pos[1]]
                res = kg.search_triple(entity_i_qid, entity_j_qid, k_hop=1)
                if res is not None: # 说明当前两个实体存在关系
                    relation_name, relation_pid = res[0][1], res[1][1]
                    relation_name_tokens = tokenizer.encode(relation_name, add_special_tokens=False)
                    # template e.g., [X] is a [Y] .
                    # template ids e.g. [685, 55, 60, 318, 257, 685, 56, 60, 764]
                    # template ids str e.g. "685 55 60 318 257 685 56 60 764"
                    template_ids_str = "{} {} {}".format(trigger_x_ids_str, " ".join(map(str, relation_name_tokens)), trigger_y_ids_str)
                    if relation_pid in relation_pid2template_ids_str.keys():
                        template_ids_str = relation_pid2template_ids_str[relation_pid]
                    prompt_ids_str = template_ids_str.replace(trigger_x_ids_str, " ".join(map(str, entity_i_tokens))).replace(trigger_y_ids_str, " ".join(map(str, entity_j_tokens)))

                    break

            pass
        else: # 默认的causal lm
            cur_prompt += token_ids + newline # 198表示\n
            a_prefix = []
            l_str = []
            cur_label_mask += [1] * len(token_ids) + [0] * len(newline)

        cur_prompt += a_prefix + l_str + newlines # 表示连续的两个\n
        cur_label_mask += [0] * len(a_prefix) + [1] * len(l_str) + [0] * len(newlines)
        assert len(cur_prompt) == len(cur_label_mask)

        if (len(cur_prompt) + len(prompt)) > max_len:

            final_label_mask = list()
            if only_train_last and len(label_mask) > 1: # 如果只允许最后一个in-context example计算loss，则前面的所有example的label mask全部置0
                for lm in label_mask[:-1]:
                    final_label_mask.extend([0] * len(lm))
            elif not only_train_last and len(label_mask) > 1:
                for lm in label_mask[:-1]:
                    final_label_mask.extend(lm)
            final_label_mask.extend(label_mask[-1]) # 最后一个in-context example一定需要计算loss

            all_prompt_data.append({
                "token_ids": prompt,
                "label_masks": final_label_mask
            })
            prompt = list()
            label_mask = list()
        prompt.extend(cur_prompt)
        # label_mask.extend(cur_label_mask)
        label_mask.append(cur_label_mask)

    return all_prompt_data

if __name__ == "__main__":
    """
    本脚本用于生成language modeling feature数据
    python3 ./processor/pretraining/kg_enhance_causal_lm/generate_lm_data.py
    """
    # === 数据生成控制
    only_train_last = True
    data_mode = "only_mep"
    assert data_mode in ["mixed", "only_mep", "only_ecg", "only_rkp", "only_causal_lm"]

    if data_mode == "only_mep":
        save_dir = "/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_pretrain_prompt_data_gpt_mep"
    elif data_mode == "only_ecg":
        save_dir = "/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_pretrain_prompt_data_gpt_ecg"
    elif data_mode == "only_rkp":
        save_dir = "/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_pretrain_prompt_data_gpt_rkp"
    elif data_mode == "mixed":
        save_dir = "/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_pretrain_prompt_data_gpt"
    if only_train_last is True:
        save_dir = "{}_only_train_last".format(save_dir)
    print("save_dir=", save_dir)

    # with open("/Users/wangjianing/Desktop/开源代码与数据模型/test/total_pretrain_data_gpt.json", "r", encoding="utf-8") as fr:
    with open("/wjn/nlp_task_datasets/kg-pre-trained-corpus/total_pretrain_data_gpt/data.json", "r", encoding="utf-8") as fr:
        lines = json.load(fr)
    # lines = [json.loads(line.replace("\n", "")) for line in tqdm(lines)]
    print("finish loading.")
    shuffle(lines)
    print("finish shuffle.")
    data_num = len(lines)
    mep_data, egc_data, origin_data = lines[: int(data_num * 0.33)], lines[int(data_num * 0.33): int(data_num * 0.67)], lines[int(data_num * 0.67):]
    data_type = [mep_data, egc_data, origin_data]
    # Masked Entity Prediction (MEP)
    params1 = {
        "max_len": 1016,
        "q_prefix": "",
        "a_prefix": "",
        "task_type": "mep"
    }
    # Entity Condictional Generation (ECG)
    params2 = {
        "max_len": 1016,
        "q_prefix": " Entities: ",
        "a_prefix": " Text: ",
        "task_type": "ecg"
    }
    # Relational Knowledge Prediction (RKP)
    params3 = {
        "max_len": 1016,
        "q_prefix": "",
        "a_prefix": "",
        "task_type": "rkp"
    }
    # original causal lm
    params4 = {
        "max_len": 1016,
        "q_prefix": "",
        "a_prefix": "",
        "task_type": ""
    }
    if data_mode == "mixed":
        param_type = [params1, params2, params3] # mixed
    elif data_mode == "only_mep":
        param_type = [params1, params1, params1] # only MEP task
    elif data_mode == "only_ecg":
        param_type = [params2, params2, params2] # only ECG task
    elif data_mode == "only_rkp":
        param_type = [params3, params3, params3] # only RKP task
    elif data_mode == "only_causal_lm":
        param_type = [params4, params4, params4] # only causal lm task

    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
    all_prompt_data = list()
    for data, param in zip(data_type, param_type):
        kg = None
        if param["task_type"] in ["rkp"]:
            kg = Wikidata5m("/wjn/nlp_task_datasets/wikidata5m", tokenizer=tokenizer)
        prompt_data = construct_prompt_token(param, tokenizer, data, kg=kg, only_train_last=only_train_last)
        all_prompt_data.extend(prompt_data)
        # print("prompt[0]=", all_prompt_data[0])
        # print("prompt[0].text=", tokenizer.decode(all_prompt_data[0]["token_ids"]))
    print("len(all_prompt_data)=", len(all_prompt_data))
    # shuffle(all_prompt_data)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "prompt_data.json"), "w", encoding="utf-8") as fw:
        for line in tqdm(all_prompt_data):
            fw.write("{}\n".format(json.dumps(line, ensure_ascii=False)))
    # with open("./total_pretrain_data_gpt/prompt_data.json", "w", encoding="utf8") as fw:
    #     json.dump(all_prompt_data, fw)

    """
    e.g.
    {"token_ids": [14539, 871, 25, 220, 6929, 952, 509, 505, 87, 13, 198, 8206, 25, 220, 1375, 373, 3706, 366, 39, 34655, 22825, 310, ],
    "label_masks": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    """
