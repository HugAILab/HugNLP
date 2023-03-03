"""
用于读取wikidata知识图谱
download wikidata5m: https://deepgraphlearning.github.io/project/wikidata5m
"""

from lib2to3.pgen2 import token
import logging
import sys
import os
import json
import random
import tagme
from tqdm import tqdm
from typing import Dict, List, Union

# 标注的“Authorization Token”，需要注册才有
tagme.GCUBE_TOKEN = "714a4c40-7452-4628-9527-df1274f157cb-843339462"

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")

class Wikidata5m:
    def __init__(
        self,
        data_dir: str,
        load_entity_only: bool = False,
        load_relation_only: bool = False,
        tokenizer = None
        ) -> None:
        self.data_dir = data_dir
        if load_entity_only:
            self.entity_qid2name, self.name2entity_qid = self.load_entity()
        if load_relation_only:
            self.relation_pid2name, self.name2relation_pid = self.load_relation()
            self.relation_pid2template = self.load_relation_template()
        if not load_entity_only and not load_relation_only:
            self.entity_qid2name, self.name2entity_qid = self.load_entity()
            self.relation_pid2name, self.name2relation_pid = self.load_relation()
            self.entity_subgraph = self.load_triple()
            self.relation_pid2template = self.load_relation_template()
        if tokenizer is not None:
            self.tokenizer = tokenizer


    def load_entity(self):
        """
        Q5196650	Cut Your Hair	cut your hair
        Q912600	Straumur-Burðarás	Straumur	straumur–burðarás
        Q47551	ditiano	tipciano	titiaen geovene
        return Union(Dict[str, List], Dict[str, List])
        """
        print("loading wikidata entity")
        entity_qid2name = dict()
        name2entity_qid = dict()
        with open(os.path.join(self.data_dir, "wikidata5m_entity.txt"), "r", encoding="utf-8") as fr:
            lines = fr.readlines()
        for line in tqdm(lines):
            line = line.replace("\n", "")
            entity_qid, entity_name_list = line.split("\t")[0], line.split("\t")[1:]
            # qid -> name
            if entity_qid not in entity_qid2name.keys():
                entity_qid2name[entity_qid] = entity_name_list
            else:
                entity_qid2name[entity_qid].extend(entity_name_list)
            # name -> qid
            for entity in entity_name_list:
                assert entity not in name2entity_qid.keys()
                name2entity_qid[entity] = entity_qid
        return entity_qid2name, name2entity_qid

    def load_relation(self):
        """
        P489	currency symbol description
        P834	train depot	railway depot	depot	rail yard
        P2629	BBFC rating	BBFC certificate
        P1677	index case of
        return Dict[str, Dict[str, List]]
        """
        print("loading wikidata relation")
        relation_pid2name = dict()
        name2relation_pid = dict()
        with open(os.path.join(self.data_dir, "wikidata5m_relation.txt"), "r", encoding="utf-8") as fr:
            lines = fr.readlines()
        for line in tqdm(lines):
            line = line.replace("\n", "")
            relation_pid, relation_name_list = line.split("\t")[0], line.split("\t")[1:]
            # qid -> name
            if relation_pid not in relation_pid2name.keys():
                relation_pid2name[relation_pid] = relation_name_list
            else:
                relation_pid2name[relation_pid].extend(relation_name_list)
            # name -> qid
            for relation in relation_name_list:
                # assert relation not in name2relation_pid.keys()
                name2relation_pid[relation] = relation_pid
        return relation_pid2name, name2relation_pid

    def load_relation_template(self):
        if not os.path.exists(os.path.join(self.data_dir, "relations.jsonl")):
            return {}
        with open(os.path.join(self.data_dir, "relations.jsonl"), "r", encoding="utf-8") as fr:
            lines = fr.readlines()
        relation_pid2template = dict()
        for line in lines:
            line = json.load(line)
            relation_pid2template[line["relation"]] = line["template"]
        # relation_pid2template["other"] = "[X] {} [Y] ."
        return relation_pid2template

    def load_triple(self):
        """
        Q29387131	P31	Q5
        Q326660	P1412	Q652
        Q7339549	P57	Q1365729
        Q554335	P27	Q29999
        Q20641639	P54	Q80955
        return Dict[str, Dict[str, List]]
        """
        print("loading wikidata triplet")
        entity_subgraph = dict()
        with open(os.path.join(self.data_dir, "wikidata5m_all_triplet.txt"), "r", encoding="utf-8") as fr:
            lines = fr.readlines()
        for line in tqdm(lines):
            head_qid, relation_pid, tail_qid = line.replace("\n", "").split("\t")
            if head_qid not in entity_subgraph.keys():
                entity_subgraph[head_qid] = dict()
            if relation_pid not in entity_subgraph[head_qid].keys():
                entity_subgraph[head_qid][relation_pid] = list()
            entity_subgraph[head_qid][relation_pid].append(tail_qid)
        return entity_subgraph


    def search_entity(self, entity_name):
        # 搜索某个实体是否存在，若存在返回qid
        if entity_name in self.name2entity_qid.keys():
            return self.name2entity_qid[entity_name]
        return None

    def search_subgraph(self, entity_qid, relation_pid=None):
        # 返回某个实体的1-hop子图
        if entity_qid in self.entity_subgraph.keys():
            sub_graph = self.entity_subgraph[entity_qid]
            if relation_pid is not None and relation_pid in sub_graph.keys():
                return self.entity_subgraph[entity_qid][relation_pid]
            return sub_graph
        return None

    def search_triple(self, head_qid, tail_qid, k_hop=1):
        # 给定头尾实体名称，返回对应的k-hop三元组
        # 目前仅支持1-hop
        # head_qid = self.search_entity(head_entity_name)
        # tail_qid = self.search_entity(tail_entity_name)
        head_entity_name = self.entity_qid2name[head_qid]
        tail_entity_name = self.entity_qid2name[tail_qid]
        if head_qid is None or tail_qid is None:
            return None
        sub_graph = self.search_subgraph(head_qid)
        for relation_pid in sub_graph.keys():
            relation_name = self.relation_pid2name[relation_pid]
            tail_entities = sub_graph[relation_pid]
            if tail_qid in tail_entities:
                return (head_entity_name, relation_name, tail_entity_name), (head_qid, relation_pid, tail_qid)
        return None

    def tokenize_template(self):
        trigger_x = self.tokenizer.encode(" [X]", add_special_tokens=False)
        trigger_y = self.tokenizer.encode(" [Y]", add_special_tokens=False)
        relation_pid2template_ids_str = dict()
        for relation_pid, template in self.relation_pid2template.items():
            template_ids = self.tokenizer.encode(" " + template, add_special_tokens=False)
            template_ids_str = " ".join(map(str, template_ids))
            relation_pid2template_ids_str[relation_pid] = template_ids_str
        return " ".join(map(str, trigger_x)), " ".join(map(str, trigger_y)), relation_pid2template_ids_str



class Tagme:
    def __init__(self) -> None:
        pass

    def tagme_annotation(self, txt):

        def Annotation_mentions(txt):
            """
            发现那些文本中可以是维基概念实体的概念
            :param txt: 一段文本对象，str类型
            :return: 键值对，键为本文当中原有的实体概念，值为该概念作为维基概念的概念大小，那些属于维基概念但是存在歧义现象的也包含其内
            """
            annotation_mentions = tagme.mentions(txt)
            dic = dict()
            for mention in annotation_mentions.mentions:
                try:
                    dic[str(mention).split(" [")[0]] = str(mention).split("] lp=")[1]
                except:
                    logger.error("error annotation_mention about " + mention)
            return dic


        def Annotate(txt, language="en", theta=0.1):
            """
            解决文本的概念实体与维基百科概念之间的映射问题
            :param txt: 一段文本对象，str类型
            :param language: 使用的语言 “de”为德语, “en”为英语，“it”为意语.默认为英语“en”
            :param theta:阈值[0, 1]，选择标注得分，阈值越大筛选出来的映射就越可靠，默认为0.1
            :return:键值对[(A, B):score]  A为文本当中的概念实体，B为维基概念实体，score为其得分
            """
            annotations = tagme.annotate(txt, lang=language)
            dic = dict()
            for ann in annotations.get_annotations(theta):
                # print(ann)
                try:
                    A, B, score = str(ann).split(" -> ")[0], str(ann).split(" -> ")[1].split(" (score: ")[0], str(ann).split(" -> ")[1].split(" (score: ")[1].split(")")[0]
                    dic[(A, B)] = score
                except:
                    logger.error("error annotation about " + ann)
            return dic

        # obj = Annotation_mentions(txt)
        obj = Annotate(txt, theta=0.2)
        mention_list = [i[0] for i in obj.keys()]
        entity_list = [i[1] for i in obj.keys()]
        return mention_list, entity_list


# if __name__ == "__main__":
#     # kg = Wikidata5m("./wikidata5m")
#     tgm = Tagme()
#     # kg = Wikidata5m("./")
#     # f = open("text.txt", "r", encoding="utpf8")
#     # txt = f.read()
#     txt = "The Solar Riser made the first man - carrying flight on solar power at noon on 29 April 1979 at Flabob Airport in Riverside , California ."
#     entity_list = tgm.tagme_annotation(txt)
#     print(entity_list)
#     for entity in entity_list:
#         qid = kg.search_entity(entity)
#         print("{}.qid={}".format(entity, qid))

#     pass
