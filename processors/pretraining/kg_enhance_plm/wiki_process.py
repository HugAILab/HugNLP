import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
import random
from tqdm import tqdm
from utils.tagme_test import Annotate
path = "../data_corpus/wiki"

def read_wiki(path):
    dirs = os.listdir(path=path)
    corpus = list()
    for dir in tqdm(dirs):
        sub_path = os.path.join(path, dir)
        files = os.listdir(sub_path)
        for file in files:
            file_path = os.path.join(sub_path, file)
            with open(file_path, "r", encoding="utf-8") as fr:
                lines = fr.readlines()
            for line in lines:
                line = line.replace("\n", "")
                if "http" in line:
                    continue
                tokens = line.split(" ")
                if len(tokens) < 20:
                    continue
                corpus.append(line)

    random.shuffle(corpus)
    train_corpus = corpus[: -50000]
    validation_corpus = corpus[-50000:]
    print("corpus num: {}".format(len(corpus))) # 32,715,108
    print("train corpus num: {}".format(len(train_corpus))) # 32,705,108
    print("validation corpus num: {}".format(len(validation_corpus))) # 10,000

    with open("train.txt", "w", encoding="utf-8") as fw:
        for text in tqdm(train_corpus):
            fw.write(text + "\n")

    with open("validation.txt", "w", encoding="utf-8") as fw:
        for text in tqdm(validation_corpus):
            fw.write(text + "\n")


def tag_me(path, file_name):
    new_file_name = file_name.split(".")[0] + "_with_entity." + file_name.split(".")[1]
    with open(os.path.join(path, file_name), "r", encoding="utf-8") as fr:
        lines = fr.readlines()
    with open(os.path.join(path, new_file_name), "w", encoding="utf-8") as fw:
        for line in tqdm(lines):
            txt = line.replace("\n", "")
            obj = Annotate(txt, theta=0.2)
            entities = list(set([i[1] for i in obj.keys()])) # list()
            example = "{}\t{}".format(txt, "\t".join(entities))
            fw.write(example + "\n")





if __name__ == "__main__":
    # read_wiki(path)
    # tag_me(path, "train_10000.txt")
    # tag_me(path, "train_10_percent.txt")
    tag_me(path, "train.txt")
