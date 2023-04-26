import json
from datasets import Dataset


def format_data(file_path):
    results = {
        "ID": [],
        "instruction": [],
        "target": [],
        # "raw_info": [],
    }
    with open(file_path, encoding="utf-8") as f:
        content = json.load(f)
        for sample in content:
            try:
                results["ID"].append(sample["ID"])
                results["instruction"].append(sample["instruction"])
                results["target"].append(sample["target"])
            except Exception as e:
                print(e)
                print(sample)
                input()
            # results["raw_info"].append(sample)
        print("===========SAMPLE==========")
        print(results["instruction"][0])
        print(results["target"][0])
        # print(results["raw_info"][0])
        results = Dataset.from_dict(results)
    return results
