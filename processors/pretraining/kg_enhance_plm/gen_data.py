# -*- coding: utf-8 -*-
# @Time    : 2022/03/20 11:19
# @Author  : Jianing Wang
# @Email   : lygwjn@gmail.com
# @File    : gen_data.py
# !/usr/bin/env python
# coding=utf-8

import random
import numpy as np
from transformers import RobertaTokenizer
import os
from tqdm import tqdm
import multiprocessing
import json

# 加载Wikidata5M知识库
def load_data(data_dir):
    wiki5m_alias2qid, wiki5m_qid2alias = {}, {}
    with open(os.path.join(data_dir, "wikidata5m_entity.txt"), "r",
              encoding="utf-8") as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            v = line.strip().split("\t")
            if len(v) < 2:
                continue
            qid = v[0] # entity idx
            wiki5m_qid2alias[qid] = v[1:]  # {qid: ["xx", ...]}
            for alias in v[1:]: # all entity name

                wiki5m_alias2qid[alias] = qid # {"xx xxx xx": qid}

    # d_ent = wiki5m_alias2qid
    print("wikidata5m_entity.txt (Wikidata5M) loaded!")

    wiki5m_pid2alias = {}
    with open(os.path.join(data_dir, "wikidata5m_relation.txt"), "r",
              encoding="utf-8") as fin:
        lines = fin.readlines()
        for i in tqdm(range(len(lines))):
            line = lines[i]
            v = line.strip().split("\t")
            if len(v) < 2:
                continue
            wiki5m_pid2alias[v[0]] = v[1]
    print("wikidata5m_relation.txt (Wikidata5M) loaded!")

    head_cluster, tail_cluster = {}, {} # {"entity_id": [(relation_id, entity_id), ...]}
    total = 0
    for triple_file in [
        "wikidata5m_transductive_train.txt",
        "wikidata5m_transductive_valid.txt",
        "wikidata5m_transductive_test.txt"
    ]:
        with open(os.path.join(data_dir, triple_file), "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            for i in tqdm(range(len(lines))):
                line = lines[i]
                v = line.strip().split("\t")
                if len(v) != 3:
                    continue
                h, r, t = v
                # if (h, r, t) not in fewrel_triples:
                if h in head_cluster:
                    head_cluster[h].append((r, t))
                else:
                    head_cluster[h] = [(r, t)]
                if t in tail_cluster:
                    tail_cluster[t].append((r, h))
                else:
                    tail_cluster[t] = [(r, h)]
                # else:
                #     num_del += 1
                total += 1
    print("wikidata5m_triplet.txt (Wikidata5M) loaded!")
    # print("deleted {} triples from Wikidata5M.".format(num_del))

    print("entity num: {}".format(len(head_cluster.keys())))

    """
    d_ent: {"entity alias": qid}
    head_cluster: {"entity_id": [(relation_id, entity_id), ...]}
    """
    return wiki5m_alias2qid, wiki5m_qid2alias, wiki5m_pid2alias, head_cluster


# args
max_neighbors = 15
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def run_proc(
        cpu_id,
        file_list,
        wiki5m_alias2qid,
        wiki5m_qid2alias,
        head_cluster,
        min_seq_len=100,
        max_seq_len=300,
        output_folder="./pretrain_data/data/",
):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    target_filename = os.path.join(output_folder, "data_{}.json".format(cpu_id + 100))
    fout_normal = open(target_filename, "w", encoding="utf-8")
    for i in tqdm(range(len(file_list))):
        input_name = file_list[i]
        fin = open(input_name, "r", encoding="utf-8")

        for doc in fin:
            doc = doc.strip()
            segs = doc.split("[_end_]")
            content = segs[0] # 文本
            sentences = content.split(".")
            map_segs = segs[1:] # 所有mention-entity title
            # 文本中所有mention对应到Wikidata5m的实体qid
            maps = {}  # mention -> QID 文本中的mention对齐到kg
            for x in map_segs:
                v = x.split("[_map_]")
                if len(v) != 2:
                    continue
                if v[1] in wiki5m_alias2qid:  # if a wikipedia title is the alias of an entity in wikidata
                    maps[v[0]] = wiki5m_alias2qid[v[1]]
                elif v[1].lower() in wiki5m_alias2qid:
                    maps[v[0]] = wiki5m_alias2qid[v[1].lower()]
            blocks, word_lst = [], []
            # 将文本block到合适的长度区间内
            # 最大长度做采样，避免训练样本长度过于统一
            sample_min_len = random.randint(min_seq_len, max_seq_len)
            s = ""
            for sent in sentences:
                s = "{} {}.".format(s, sent)
                word_lst = tokenizer.tokenize(s)

                if len(word_lst) >= sample_min_len:
                    blocks.append(s)
                    s = ""
                if len(blocks) >= max_seq_len:
                    blocks = blocks[: max_seq_len]
            if len(s) > 0:
                blocks.append(s)
            for block in blocks:
                anchor_segs = [x.strip() for x in block.split("sepsepsep")]
                tokens, entity_ids, mention_spans = [], [], []
                for x in anchor_segs:
                    if len(x) < 1:
                        continue
                    if x in maps and maps[x]: # 如果是实体
                        start = len(tokens) + 1 # 预留CLS的位置
                        entity_tokens = tokenizer.encode(x)
                        tokens.extend(entity_tokens)
                        end = len(tokens)
                        mention_spans.append((start, end)) # 当前mention对应文本内的区间
                        entity_ids.append(maps[x]) # 当前mention对齐的知识库实体qid

                    else: # 如果只是一个文本片段
                        # words = tokenizer.tokenize(x)
                        words = tokenizer.encode(x)
                        words = words[:max_seq_len]
                        tokens.extend(words)
                if len(entity_ids) == 0: # 只保留含有实体的文本
                    continue
                data = {
                    "token_ids": tokens,
                    "entity_qid": entity_ids,
                    "entity_pos": mention_spans,
                    "relation_pid": None,
                    "relation_pos": None,
                }
                data = json.dumps(data)
                fout_normal.write("{}\n".format(data))

        fin.close()
    fout_normal.close()


class MultiProcess:
    def __init__(self, dataset=None, wiki5m_alias2qid=None, wiki5m_qid2alias=None, head_cluster=None):
        self.dataset = dataset
        self.wiki5m_alias2qid = wiki5m_alias2qid
        self.wiki5m_qid2alias = wiki5m_qid2alias
        self.head_cluster = head_cluster
        self.output_folder = "./pretrain_data/data/"


    def process(self, digits, fold="1by1"):  # 处理函数：用于处理数据
        file_list, para_id = digits

        run_proc(
            para_id,
            file_list,
            self.wiki5m_alias2qid,
            self.wiki5m_qid2alias,
            self.head_cluster,
            min_seq_len=80,
            max_seq_len=200,
            output_folder=self.output_folder,
        )

    def run(self):  # 线程分配函数
        n_cpu = multiprocessing.cpu_count()  # 获得CPU核数
        # n_cpu = 4 # 也可以人为指定
        num = len(self.dataset)  # 数据集样本数量
        self.n_cpu = n_cpu
        print("cpu num: {}".format(n_cpu))
        chunk_size = int(num / n_cpu)  # 分摊到每个CPU上的样本数量
        procs = []
        for i in range(0, n_cpu):
            min_i = chunk_size * i
            if i < n_cpu - 1:
                max_i = chunk_size * (i + 1)
            else:
                max_i = num
            digits = [self.dataset[min_i: max_i], i]
            # 每个线程唤醒并执行
            procs.append(multiprocessing.Process(target=self.process, args=(digits, "parallel")))
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()

    def merge(self):  # 数据合并函数：对每个线程上的处理好的数据进行合并
        for path, _, filenames in os.walk(self.output_folder):
            for filename in filenames:
                file_list.append(os.path.join(path, filename))
        fw = open(os.path.join(self.output_folder, "data.json"), "w", encoding="utf-8")
        print("Start merging ...")
        for file in tqdm(file_list):
            with open(file, "r", encoding="utf-8") as fr:
                lines = fr.readlines()
            for line in lines:
                fw.write(line)
        print("Meger is done.")




if __name__ == "__main__":
    print("load from wikidata5m")
    kg_output = "./pretrain_data/kg/"
    wiki5m_alias2qid, wiki5m_qid2alias, wiki5m_pid2alias, head_cluster = load_data(kg_output)
    if not os.path.exists(kg_output):
        os.makedirs(kg_output)
    # 保存KG信息
    np.savez(
        os.path.join(kg_output, "wiki_kg.npz"),
        wiki5m_alias2qid=wiki5m_alias2qid,
        wiki5m_qid2alias=wiki5m_qid2alias,
        wiki5m_pid2alias=wiki5m_pid2alias,
        head_cluster=head_cluster,
    )
    kg = np.load(
        os.path.join(kg_output, "wiki_kg.npz"), allow_pickle=True
    )
    wiki5m_alias2qid, wiki5m_qid2alias, head_cluster = kg["wiki5m_alias2qid"][()], kg["wiki5m_qid2alias"][()], kg["head_cluster"][()]

    input_folder = "./pretrain_data/ann"
    file_list = []
    for path, _, filenames in os.walk(input_folder):
        for filename in filenames:
            file_list.append(os.path.join(path, filename))
    print("# of files", len(file_list))

    m = MultiProcess(
        dataset=file_list,
        wiki5m_alias2qid=wiki5m_alias2qid,
        wiki5m_qid2alias=wiki5m_qid2alias,
        head_cluster=head_cluster,
    )
    m.run()  # 多线程
    m.merge()
