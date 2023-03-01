# -*- coding: utf-8 -*-
# @Time    : 2022/03/18 18:44
# @Author  : Jianing Wang
# @Email   : lygwjn@gmail.com
# @File    : extract.py
# !/usr/bin/env python
# coding=utf-8

from bs4 import BeautifulSoup
from urllib import parse
import os
import multiprocessing
from tqdm import tqdm


class MultiProcess:
    def __init__(self, dataset=None):
        self.dataset = dataset

    def process(self, digits, fold='1by1'):  # 处理函数：用于处理数据
        file_list, para_id = digits
        for i in tqdm(range(len(file_list))):
            input_name = file_list[i]
            target = input_name.replace('pretrain_data/output',
                                        'pretrain_data/ann')
            folder = '/'.join(target.split('/')[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)
            soup = BeautifulSoup(open(input_name, encoding='utf-8'),
                                 features='html5lib')
            docs = soup.find_all('doc')
            fout = open(target, 'w', encoding='utf-8')

            for doc in docs:
                # 将原始文本去除空格空行
                content = doc.get_text(' sepsepsep ')
                try:
                    while content[0] == '\n':
                        content = content[1:]
                except:
                    continue
                content = [x.strip() for x in content.split('\n')]
                content = ''.join(content[1:])
                # 寻找所有文本中<a href=''></a>标签，其对应一个实体
                lookup = []  # 所有超链接对应的实体
                for x in doc.find_all('a'):
                    if x.get('href') is not None:
                        lookup.append((x.get_text().strip(),
                                       parse.unquote(x.get('href'))))
                # 在content后拼接所有实体，每个实体以[_end_]为分割
                # [_end_] mention [_map_] entity [_end_]
                lookup = '[_end_]'.join(['[_map_]'.join(x) for x in lookup])
                fout.write(content + '[_end_]' + lookup + '\n')

            fout.close()

    def run(self):  # 线程分配函数
        n_cpu = multiprocessing.cpu_count()  # 获得CPU核数
        num = len(self.dataset)  # 数据集样本数量
        self.n_cpu = n_cpu
        print('cpu num: {}'.format(n_cpu))
        chunk_size = int(num / n_cpu)  # 分摊到每个CPU上的样本数量
        procs = []
        for i in range(0, n_cpu):
            min_i = chunk_size * i
            if i < n_cpu - 1:
                max_i = chunk_size * (i + 1)
            else:
                max_i = num
            digits = [self.dataset[min_i:max_i], i]
            # 每个线程唤醒并执行
            procs.append(
                multiprocessing.Process(target=self.process,
                                        args=(digits, 'parallel')))
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()

    def merge(self):  # 数据合并函数：对每个线程上的处理好的数据进行合并
        pass


if __name__ == '__main__':

    input_folder = './pretrain_data/output'
    file_list = []
    for path, _, filenames in os.walk(input_folder):
        for filename in filenames:
            file_list.append(os.path.join(path, filename))
    print(len(file_list))

    m = MultiProcess(dataset=file_list)
    m.run()  # 多线程
