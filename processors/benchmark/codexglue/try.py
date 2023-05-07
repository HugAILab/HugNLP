from datasets import load_dataset
raw_datasets=load_dataset("nchen909/devign-processed")
print(raw_datasets["train"]["label"])
print(raw_datasets["train"].features["label"].names)
