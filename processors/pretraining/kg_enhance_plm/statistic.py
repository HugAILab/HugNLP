import os
import json
import pickle
from tqdm import tqdm

file_list = []
for path, _, filenames in os.walk('../pretrain_data/data'):
    for filename in filenames:
        file_list.append(os.path.join(path, filename))

ent_freq = {'<unk>': 0, '<pad>': 0, '<mask>': 0}
rel_freq = {'<unk>': 0, '<pad>': 0, '<mask>': 0}
ent_vocab = {'<unk>': 0, '<pad>': 1, '<mask>': 2}
rel_vocab = {'<unk>': 0, '<pad>': 1, '<mask>': 2}
for i in tqdm(range(len(file_list))):
    with open(file_list[i], 'r', encoding='utf-8') as fin:
        for x in fin:
            ins = json.loads(x)
            for node in ins['nodes']:
                if isinstance(node, str):
                    if node.startswith('Q'):
                        if node not in ent_freq:
                            ent_freq[node] = 1
                        else:
                            ent_freq[node] += 1
                        if node not in ent_vocab:
                            ent_vocab[node] = len(ent_vocab)
                    if node.startswith('P'):
                        if node not in rel_freq:
                            rel_freq[node] = 1
                        else:
                            rel_freq[node] += 1
                        if node not in rel_vocab:
                            rel_vocab[node] = len(rel_vocab)

with open('../read_rel_freq.bin', 'wb') as fout:
    pickle.dump(rel_freq, fout)
with open('../read_rel_vocab.bin', 'wb') as fout:
    pickle.dump(rel_vocab, fout)
print(len(rel_vocab))

with open('../read_ent_freq.bin', 'wb') as fout:
    pickle.dump(ent_freq, fout)
with open('../read_ent_vocab.bin', 'wb') as fout:
    pickle.dump(ent_vocab, fout)
print(len(ent_vocab))
