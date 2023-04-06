from transformers import PLBartTokenizer, PLBartForSequenceClassification, PLBartConfig, PLBartForConditionalGeneration
import torch
# 加载预训练模型和分词器
model_name = "/root/autodl-tmp/code/CodePrompt/save_models/clone/plbart/ckpt_test/"#"/root/autodl-tmp/CodePrompt/data/huggingface_models/plbart-base/"#
ckpt= model_name #"/root/autodl-tmp/code/CodePrompt/save_models/clone/plbart/checkpoint-best-f1"

code1 = """
def fact(n):
    return 1 if n == 0 else n * fact(n - 1)
"""
    # """
# def factorial(n):
#     if n == 0:
#         return 1
#     else:
#         return n * factorial(n-1)
# """
code2 = """
def fact(n):
    return 1 if n == 0 else n * fact(n - 1)
"""


# 加载预训练模型和分词器
model = PLBartForSequenceClassification.from_pretrained(model_name)
model_ckpt = PLBartForSequenceClassification.from_pretrained(ckpt)
tokenizer = PLBartTokenizer.from_pretrained(model_name)
config =PLBartConfig.from_pretrained(model_name)

torch.manual_seed(1234)
params1 = [name for name, param in list(model.named_parameters())]
params2 = [name for name, param in list(model_ckpt.named_parameters())]

diff=list(set(params1)^set(params2))
print(diff)
# 将两段代码拼接并用 <sep> 分隔
input_sequence = code1 + " <sep> " + code2

# 使用分词器对输入序列进行编码
inputs = tokenizer(input_sequence, return_tensors="pt")

# 将编码后的输入传递给模型
outputs = model(**inputs)

# 获取分类概率
probs = torch.softmax(outputs.logits, dim=-1)

# 获取克隆概率
clone_prob = probs[0, 1].item()

print("Clone Probability:", clone_prob)
