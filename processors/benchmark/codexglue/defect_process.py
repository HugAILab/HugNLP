
'''
# -*- coding: utf-8 -*-
Author: nchen909 NuoChen
Date: 2023-05-06 16:11:10
FilePath: /HugNLP/processors/benchmark/codexglue/defect_process.py
'''
path = '/root/autodl-tmp/HugCode/data/defect/valid.jsonl'
new_path = '/root/autodl-tmp/HugCode/data/defectnewest/valid.jsonl'
import json
import tqdm
from datasets import Dataset, DatasetDict
import json
def process_clone_examples(path):
    """Read examples from path."""
    index_path = path

    processed_lines = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data={}
            old_line=line
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            data["label"] = js['target']
            data["func1"]=code
            data["id"]=js['idx']
            processed_lines.append(json.dumps(data))
            idx += 1
    with open(new_path, "w") as outfile:
        for line in processed_lines:
            outfile.write(line + "\n")
    return 0



process_clone_examples(path)
