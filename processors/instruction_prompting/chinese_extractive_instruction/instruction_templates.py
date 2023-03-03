import json
import os.path
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from processors.instruction_prompting.chinese_extractive_instruction.instruction_prompts import *
from tqdm import tqdm
import random

dataset2instruction = {
    "cls": {
        "prompt": "文章的类别是什么？{}。【{}】",
        "keys_order": ["verbalizer", "text_a"],
        "instruction": ClassificationInstruction,
        "data_type": "classification",
    },
    "sentiment": {
        "prompt": "文章的情感态度是什么？{}。【{}】",
        "keys_order": ["verbalizer", "text_a"],
        "instruction": ClassificationInstruction,
        "data_type": "classification",
    },
    "news": {
        "prompt": "新闻的类别是什么？{}。【{}】",
        "keys_order": ["verbalizer", "text_a"],
        "instruction": ClassificationInstruction,
        "data_type": "classification",
    },
    "intent": {
        "prompt": "这句话的意图是什么？{}。【{}】",
        "keys_order": ["verbalizer", "text_a"],
        "instruction": ClassificationInstruction,
        "data_type": "classification",
    },
    "nli": {
        "prompt": "以下两句话的逻辑关系是什么？{}。【{}】和【{}】",
        "keys_order": ["verbalizer", "text_a", "text_b"],
        "instruction": NLIInstruction,
        "data_type": "classification",
    },
    "similarity": {
        "prompt": "以下两句话的内容是否相似？{}。【{}】和【{}】",
        "keys_order": ["verbalizer", "text_a", "text_b"],
        "instruction": STSInstruction,
        "data_type": "classification",
    },
    "mrc": {
        "prompt": "找到问题【{}】的答案？文章:【{}】",
        "keys_order": ["question", "context"],
        "instruction": MRCInstruction,
        "data_type": "mrc",
    },
    "ner": {
        "prompt": "找到文章中所有【{}】类型的实体？文章:【{}】",
        "keys_order": ["entity_type", "context"],
        "instruction": NERInstruction,
        "data_type": "ner",

    },
    "yesno": {
        "prompt": "问题【{}】？{}。文章:【{}】",
        "keys_order": ["question", "verbalizer", "context"],
        "instruction": YesNoInstruction,
        "data_type": "yesno",
    },
    "summ": {
        "prompt": "文章的摘要是什么？文章:【{}】",
        "keys_order": ["passage"],
        "instruction": SUMMInstruction,
        "data_type": "summ",
    },
    "keys": {
        "prompt": "找到文章的关键词？文章:【{}】",
        "keys_order": ["text_a"],
        "instruction": KEYSInstruction,
        "data_type": "keys",
    },
    "cslkeys": {
        "prompt": "下列词组可以作为文章的关键词吗？{}。关键词：【{}】。文章:【{}】",
        "keys_order": ["verbalizer", "text_b", "text_a"],
        "instruction": ClassificationInstruction,
        "data_type": "keys",
    },
    "wsc": {
        "prompt": "文章中【{}】的是【{}】吗？{}。文章:【{}】",
        "keys_order": ["target/span2_text", "target/span1_text", "verbalizer", "text"],
        "instruction": WSCInstruction,
        "data_type": "classification",
    },
    "multichoice": {
        "prompt": "阅读文章回答问题【{}】应选择:{}。文章:【{}】",
        "keys_order": ["question", "choice", "context"],
        "instruction": MultiChoiceInstruction,
        "data_type": "classification",
    },
    "chid": {
        "prompt": "文章中空缺处的成语应选择:{}。文章:【{}】",
        "keys_order": ["choice", "context"],
        "instruction": MultiChoiceInstruction,
        "data_type": "classification",
    },

}



def instruction_format(data_info: Dict, data_type, data_name) -> List[Dict]:
    instruction_data = list()
    label_mappings = data_info.get("label_mappings")
    data_list = data_info["data_list"]
    # for instruction in [dataset2instruction, dataset2instruction2, dataset2instruction3]:
    for instruction in [dataset2instruction]:
        format_info = instruction[data_type]
        instruction_processor = format_info["instruction"](
            data_name,
            data_list,
            label_mappings,
            format_info["prompt"],
            format_info["keys_order"],
            format_info["data_type"]
        )
        instruction_data.extend(instruction_processor.transform2instruction())
        # print(instruction_data[-1])

    return instruction_data


if __name__ == "__main__":
    # # 自定义将对应数据集的格式转换为prompt形式：
    # base_path = "/wjn/competition/clue/datasets/cpic/"
    # for ori_file, out_file in [["my_opend.json", "my_opend_train.json"],
    #                            ["train_data.json", "train_two_prompt.json"],
    #                            ["test_data_A.json", "test.json"],
    #                            ["test_data_B.json", "test_b.json"]
    #                            ]:
    #     data = json.load(open(os.path.join(base_path, ori_file), encoding="utf8"))
    #     out = instruction_format(data)
    #     with open(os.path.join(base_path, out_file), "w", encoding="utf8") as fout:
    #         json.dump(out, fout, indent=2, ensure_ascii=False)
    base_path = "./"
    all_instruction_data = list()
    type2data = {
        "similarity": [
            "afqmc.json", "bq_corpus.json", "chip-cdn.json", "lcqmc.json", "oppo.json", "paws-x-zh.json",
            "sohu_sts_a_ll.json", "sohu_sts_a_sl.json", "sohu_sts_a_ss.json", "sohu_sts_b_ll.json", "sohu_sts_b_sl.json",
            "sohu_sts_b_ss.json"
            ],
        "news": ["baidu_news.json", "ccfbdci2020.json", "chinanews.json", "ifeng.json", "thucnews.json"],
        "multichoice": ["c3.json"],
        "mrc": [
            "cail.json", "ccm_qa.json", "cmrc2018.json", "drcd.json", "dureader_checklist.json", "dureader_robust.json",
            "squad.json"
            ],
        "yesno": ["cail_yesno.json"],
        "sentiment": [
            "car_emotion.json", "chnsenticorp.json", "covid_emotion.json", "jd_binary_waimai_10k.json", "jd_full.json",
            "nlpcc14-sc.json", "online_shopping_10_cats.json", "simplifyweibo_4_moods.json", "turing_emotion.json",
            "weibo_senti_100k.json", "yf_amazon.json", "yf_dianping.json"
            ],
        "cls": ["catslu_traindev.json", "iflytek_ltc.json", "kuake-qic.json"],
        "chid": ["chid.json"],
        "wsc": ["cluewsc.json"],
        "nli": ["cmnli.json", "ocnli.json"],
        "intent": ["nlpcc2018_slu.json"]
    }

    for data_type, files in tqdm(type2data.items()):
        for file in files:
            data_name = file.split(".")[0]
            data = json.load(open(os.path.join(base_path, file), encoding="utf8"))
            instruction_data = instruction_format(data, data_type, data_name)
            all_instruction_data.extend(instruction_data)
    print("num={}".format(len(all_instruction_data)))
    random.shuffle(all_instruction_data)
    train_data = all_instruction_data[:2410000]
    dev_data = all_instruction_data[2410000: 2415000]
    test_data = all_instruction_data[2415000:]
    with open(os.path.join(base_path, "train_mrc_instruction.json"), "w", encoding="utf8") as fout:
        json.dump(train_data, fout, indent=2, ensure_ascii=False)
    with open(os.path.join(base_path, "dev_mrc_instruction.json"), "w", encoding="utf8") as fout:
        json.dump(dev_data, fout, indent=2, ensure_ascii=False)
    with open(os.path.join(base_path, "test_mrc_instruction.json"), "w", encoding="utf8") as fout:
        json.dump(test_data, fout, indent=2, ensure_ascii=False)
