# -*- cording: utf-8 -*-

from itertools import chain
from collections import defaultdict
import numpy as np

class Str2idx():
    __MAX_COUNT = 5000
    __MIN_COUNT = 1
    __IGNORE = -1

    def __init__(self, data):
        # label to id
        self.l_dic = defaultdict(int)
        for i, v in enumerate(set(data["label"])):
            self.l_dic[v] = i
        # word to id
        words = list(chain.from_iterable([i.split() for i in data["text"].tolist()]))
        w_freq = defaultdict(int)
        for w in words:
            w_freq[w] += 1
        w_freq = {w: f for w, f in w_freq.items() if self.__MIN_COUNT <= f <= self.__MAX_COUNT}

        self.w_dic = defaultdict(int)
        for i, v in enumerate(w_freq.keys()):
            self.w_dic[v] += i + 1

        # sentence max
        self.max_l = max(len(i.split()) for i in data["text"].tolist())

    def padding(self, arr):
        n = self.max_l - len(arr)
        for _ in range(n):
            arr.append(self.__IGNORE)
        return arr

    def doc2ids(self, doc, pad):
        doc = doc.split(" ")
        idx = [self.w_dic[d] for d in doc]
        if pad:
            idx = idx[:self.max_l]
            idx = self.padding(idx)
        return np.array(idx, dtype=np.int32)

    def label2id(self, lab):
        return np.array(self.l_dic[lab], dtype=np.int32)

    def __call__(self, data, pad):
        return [(self.label2id(t), self.doc2ids(x, pad)) for t, x in zip(data["label"], data["text"])]
