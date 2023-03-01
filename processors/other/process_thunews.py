import os
import random
from tqdm import tqdm
path = '/wjn/clue/datasets/CLUEdatasets/THUCNews'
topics = os.listdir(path)
corpus = list()
for topic in tqdm(topics):
    sub_path = os.path.join(path, topic)
    if os.path.isfile(sub_path):
        continue
    files = os.listdir(sub_path)
    for file in files:
        with open(os.path.join(sub_path, file), 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
        for line in lines:
            line = line.strip()
            line = line.replace('\n', '').replace(' ', '')
            if len(line) < 10:
                continue
            corpus.append(line)

print('num=', len(corpus))  # 843w
random.shuffle(corpus)
# sample 80w
corpus = corpus[:800000]
train_corpus = corpus[:-10000]
dev_corpus = corpus[-10000:]

with open(os.path.join(path, 'train.txt'), 'w', encoding='utf-8') as fw:
    for line in tqdm(train_corpus):
        fw.write(line + '\n')

with open(os.path.join(path, 'dev.txt'), 'w', encoding='utf-8') as fw:
    for line in tqdm(dev_corpus):
        fw.write(line + '\n')
