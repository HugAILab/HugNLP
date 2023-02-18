import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
import random
from transformers import RobertaTokenizer
import multiprocessing
import json
from typing import List
from tqdm import tqdm

from processors.pretraining.kg_enhance_plm.kg_prompt import KGPrompt

null = None

def run_proc(
    para_id: int,
    examples: List,
    kg_prompt: KGPrompt,
    tokenizer: RobertaTokenizer,
    output_folder: str
):
    fw = open(os.path.join(output_folder, "feature_{}.json".format(para_id)), 'w', encoding='utf-8')

    ei = 0
    for example in tqdm(examples):
        example = eval(example)
        rd = random.random()
        if rd <= 0.34:
            task = 1
        elif rd >= 0.67:
            task = 3
        else:
            task = 2
        ei += 1
        # step1 融合kg demonstration，
        is_negative = False if random.random() < 0.8 else True
        if task != 1:
            is_negative = False
        prompt = kg_prompt.get_demonstration(example, is_negative, start_from_input=False)
        '''
        prompt = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'noise_detect_label': 0 if is_negative else 1,
            'entity_spans': entity_spans,
            'relation_spans': relation_spans,
            'token_type_span': token_type_span
        }
        '''
        # text, entity_ids, mention_spans = example['text'], example['entity_ids'], example['mention_spans']
        # step2: padding ANd token type
        input_ids, token_type_ids, entity_spans, relation_spans, token_type_span = \
            prompt['input_ids'], prompt['token_type_ids'], prompt['entity_spans'], \
            prompt['relation_spans'], prompt['token_type_span']
        kg_prompt_ids = input_ids[token_type_span[0][1]:]
        input_ids = input_ids[: token_type_span[0][1]]

        if len(entity_spans) == 0 or len(relation_spans) == 0:
            continue

        # mlm_labels = [-100] * len(input_ids)
        # entity_label = None
        # entity_negative = None
        # relation_label = None
        # relation_negative = None

        if task == 1:
            # MLM mask采样
            # 只对context部分进行mlm采样。15%的进行mask，其中80%替换为<MASK>，10%随机替换其他词，10%保持不变

            feature = {
                'input_ids': input_ids,
                'kg_prompt_ids': kg_prompt_ids,
                'noise_detect_label': prompt['noise_detect_label'],
                'task_id': task,
            }
        elif task == 2:
            # entity prediction
            # 在demonstration部分随机挑选一个实体，并替换为mask
            max_len = 20
            entity_span = random.sample(entity_spans, 1)[0]
            start, end = entity_span
            if end - start > max_len - 2:
                end = start + max_len - 2
            entity_label = [tokenizer.cls_token_id] + kg_prompt_ids[start: end] + [tokenizer.sep_token_id] \
                           + [tokenizer.pad_token_id] * (max_len - (end - start) - 2)
            _, entity_negative = kg_prompt.sample_entity(neg_num=5)
            entity_negative = kg_prompt.encode_kg(entity_negative, max_len=max_len)
            # mlm_labels[start: end] = input_ids[start: end]
            kg_prompt_ids[start: end] = [tokenizer.mask_token_id] * (end - start)

            feature = {
                'input_ids': input_ids,
                'kg_prompt_ids': kg_prompt_ids,
                # 'mlm_labels': mlm_labels,
                'entity_label': entity_label,
                'entity_negative': entity_negative,
                'task_id': task,
            }

        elif task == 3:
            max_len = 5
            relation_span = random.sample(relation_spans, 1)[0]
            start, end = relation_span
            if end - start > max_len - 2:
                end = start + max_len - 2
            relation_label = [tokenizer.cls_token_id] + kg_prompt_ids[start: end] + [tokenizer.sep_token_id] \
                             + [tokenizer.pad_token_id] * (max_len - (end - start) - 2)
            _, relation_negative = kg_prompt.sample_relation(neg_num=5)
            relation_negative = kg_prompt.encode_kg(relation_negative, max_len=max_len)
            # mlm_labels[start: end] = input_ids[start: end]
            kg_prompt_ids[start: end] = [tokenizer.mask_token_id] * (end - start)

            feature = {
                'input_ids': input_ids,
                'kg_prompt_ids': kg_prompt_ids,
                # 'mlm_labels': mlm_labels,
                'relation_label': relation_label,
                'relation_negative': relation_negative,
                'task_id': task,
            }

        fw.write("{}\n".format(json.dumps(feature)))
    fw.close()


class MultiProcess:
    def __init__(self, kg_prompt: KGPrompt, tokenizer: RobertaTokenizer, dataset: List, output_folder: str):
        self.kg_prompt = kg_prompt
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.output_folder = output_folder


    def process(self, digits, fold="1by1"):  # 处理函数：用于处理数据
        examples, para_id = digits

        run_proc(
            para_id,
            examples,
            self.kg_prompt,
            self.tokenizer,
            output_folder=self.output_folder,
        )

    def run(self):  # 线程分配函数
        # self.n_cpu = multiprocessing.cpu_count()  # 获得CPU核数
        self.n_cpu = 32
        num = len(self.dataset)  # 数据集样本数量
        print('cpu num: {}'.format(self.n_cpu))
        chunk_size = int(num / self.n_cpu)  # 分摊到每个CPU上的样本数量
        procs = []
        for i in range(0, self.n_cpu):
            min_i = chunk_size * i
            if i < self.n_cpu - 1:
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
        file_list = ["feature_{}.json".format(i) for i in range(self.n_cpu)]
        fw = open(os.path.join(self.output_folder, 'data2kw_2.6kw.json'), 'w', encoding='utf-8')
        print('Start merging ...')
        for file in tqdm(file_list):
            with open(os.path.join(self.output_folder, file), 'r', encoding='utf-8') as fr:
                lines = fr.readlines()
            for line in lines:
                fw.write(line)
        print("Meger is done.")

def merge(output_folder, n_cpu):  # 数据合并函数：对每个线程上的处理好的数据进行合并
    file_list = ["feature_{}.json".format(i) for i in range(n_cpu)]
    fw = open(os.path.join(output_folder, 'data2kw_2.6kw.json'), 'w', encoding='utf-8')
    print('Start merging ...')
    num = 0
    for file in tqdm(file_list):
        with open(os.path.join(output_folder, file), 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
        num += len(lines)
        for line in lines:
            fw.write(line)
    fw.close()
    print("Meger is done.")
    print("Total {}".format(num))

if __name__ == "__main__":
    input_folder = './pretrain_data/data/'
    output_folder = './pretrain_data/features/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    kg_prompt = KGPrompt(tokenizer=tokenizer)
    with open(os.path.join(input_folder, 'total_pretrain_data.json'), 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    print('len(lines)=', len(lines))
    lines = lines[20000000:]
    # lines = [eval(line) for line in tqdm(lines)]
    # random.shuffle(lines)
    # rands = [random.random() for i in range(len(lines))]

    m = MultiProcess(
        kg_prompt=kg_prompt,
        tokenizer=tokenizer,
        dataset=lines,
        output_folder=output_folder,
    )
    m.run()  # 多线程
    m.merge()



    # merge(output_folder, 50)

