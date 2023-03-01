from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import numpy as np
import random


def load_data_and_draw(use_sample=False):
    '''
    @Param use_sample: 是否对O类别进行采样
    '''
    path = 'prototype_9.npy'
    digits = np.load(path, allow_pickle=True)[()]
    data, target = [], []
    color_dict = {
        0: 'r',
        1: 'k',
        2: 'b',
        3: 'g',
        4: 'm',
        5: 'c',
        6: '#ccc',
        7: '#22cc88',
        8: 'orange',
        9: 'brown',
        10: 'pink'
    }
    print("len(digits['data'])=", len(digits['data']))
    print("len(digits['target'])=", len(digits['target']))
    origin_data = digits['data'].tolist()
    data_len = len(digits['data'])
    origin_target = digits['target'][-data_len:].tolist()
    print(data_len)
    print(len(target))
    for ei, i in enumerate(origin_target):
        if i != -1:
            draw = True
            if use_sample and i == 0:
                if random.random() >= 0.7:
                    draw = False
            if draw:
                # print(i)
                target.append(color_dict[i])
                data.append(origin_data[ei])
    print(target)
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(data)
    # X_pca = PCA(n_components=2).fit_transform(digits.data)

    ckpt_dir = 'images'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    plt.figure(figsize=(5, 5))
    plt.title('{}[use_sample={}]'.format(path, str(use_sample)))
    # plt.subplot(121)
    plt.scatter(X_tsne[:, 0],
                X_tsne[:, 1],
                s=[8] * len(target),
                c=target,
                label='t-SNE',
                cmap='Oranges')
    plt.legend()
    # plt.subplot(122)
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits['target'], label="PCA")
    # plt.legend()
    plt.savefig('images/inter-5-1.png', dpi=120)
    plt.show()


if __name__ == '__main__':
    load_data_and_draw(False)
    # digits = load_digits()
    # X_tsne = TSNE(n_components=2,random_state=33).fit_transform(digits.data)
    # X_pca = PCA(n_components=2).fit_transform(digits.data)
    #
    # ckpt_dir="images"
    # if not os.path.exists(ckpt_dir):
    #     os.makedirs(ckpt_dir)
    #
    # plt.figure(figsize=(10, 5))
    # plt.subplot(121)
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target,label="t-SNE")
    # plt.legend()
    # plt.subplot(122)
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target,label="PCA")
    # plt.legend()
    # plt.savefig('images/inter-5-1.png', dpi=120)
    # plt.show()
