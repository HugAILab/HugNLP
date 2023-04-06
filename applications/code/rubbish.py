import torch
from transformers import AutoModel, PLBartTokenizer#PLBartForConditionalGeneration, PLBartTokenizer

def code_clone_detection(model, tokenizer, code1, code2):
    inputs1 = tokenizer(code1, return_tensors="pt")
    inputs2 = tokenizer(code2, return_tensors="pt")

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
    print(outputs1.keys(),outputs2.keys())
    # similarity = torch.nn.functional.cosine_similarity(outputs1.logits, outputs2.logits, dim=-1)
    similarity = torch.nn.functional.cosine_similarity(outputs1.last_hidden_state[:, 0], outputs2.last_hidden_state[:, 0])
    print(outputs1.last_hidden_state[:, 0].shape,outputs2.last_hidden_state[:, 0].shape)
    return similarity.item()

# Load pre-trained PLBart model and tokenizer
model_name = "/root/autodl-tmp/CodePrompt/data/huggingface_models/plbart-base/"#"/root/autodl-tmp/code/CodePrompt/save_models/clone/plbart/checkpoint-best-f1"
tokenizer = PLBartTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Sample code snippets for clone detection
code1 = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""

code2 = """
def fact(n):
    return 1 if n == 0 else n * fact(n - 1)
"""

similarity = code_clone_detection(model, tokenizer, code1, code2)
print("Similarity:", similarity)
