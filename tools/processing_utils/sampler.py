# -*- coding: utf-8 -*-
# @Time    : 2021/12/2 5:41 p.m.
# @Author  : JianingWang
# @File    : sampler.py

import numpy as np
from typing import Optional

"""
random sampling for each label
"""
def random_sampling(raw_datasets, num_examples_per_label: Optional[int]=16):
    label_list = raw_datasets["label"] # [0, 1, 0, 0, ...]
    label_dict = dict()
    # denote index of each label
    for ei, label in enumerate(label_list):
        if label not in label_dict.keys():
            label_dict[label] = list()
        label_dict[label].append(ei)
    # random sample k examples of each class
    few_example_ids = list()
    for label, eid_list in label_dict.items():
        idxs = np.random.choice(len(eid_list), size=num_examples_per_label, replace=False)
        selected_eids = [eid_list[i] for i in idxs]
        few_example_ids.extend(selected_eids)
    return few_example_ids
