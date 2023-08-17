
'''
# -*- coding: utf-8 -*-
Author: nchen909 NuoChen
Date: 2023-05-06 16:11:10
FilePath: /HugNLP/processors/benchmark/codexglue/clone_process.py
'''
path = '/root/autodl-tmp/HugCode/data/clone/test.txt'
new_path = '/root/autodl-tmp/HugCode/data/clonenewest/test.jsonl'
import json
import tqdm
from datasets import Dataset, DatasetDict
import json
def process_clone_examples(path):
    """Read examples from path."""
    index_path = path
    url_to_code = {}
    with open('/'.join(index_path.split('/')[:-1]) + '/data.jsonl', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            # code_tokens, dfg = extract_dataflow(js['func'], parsers['java'], 'java')
            # code = ' '.join(code_tokens)
            # pdb.set_trace()
            url_to_code[js['idx']] = code

    processed_lines = []
    with open(index_path, encoding="utf-8") as f:
        idx = 0
        for line in f:
            data={}
            old_line = line
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data["label"] = label
            data["func1"]=url_to_code[url1]
            data["func2"]=url_to_code[url2]
            data["id"]=idx
            processed_lines.append(json.dumps(data))
            idx += 1
    with open(new_path, "w") as outfile:
        for line in processed_lines:
            outfile.write(line + "\n")
    return data



process_clone_examples(path)

# # 读取 train.jsonl 文件
# with open(new_path, "r") as infile:
#     lines = infile.readlines()

# data_list=[]
# # 将文件中的每一行转换为字典对象
# for line in lines:
#     data_list.append(json.loads(line))
# # 创建一个包含所有数据的字典，以便将其转换为 Dataset 对象
# data_dict = {"label": [item["label"] for item in data_list],"func1": [item["func1"] for item in data_list],"func2": [item["func1"] for item in data_list], "id": [item["id"] for item in data_list]}
# # 创建 DatasetDict，包含 train 数据集
# dataset = DatasetDict({"train": Dataset.from_dict(data_dict)})

# # 将数据集上传到您的 Hugging Face 帐户
# # dataset.save_to_disk("bigclonebench")
# dataset.push_to_hub("nchen909/bigclonebench-processed", private=False)
