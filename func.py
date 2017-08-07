# -*- cording: utf-8 -*-

import os
import glob
import MeCab
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from separator import Separator
from str2idx import Str2idx

SEED = 0
RATE = 0.1
SENTENCEPIECE_PATH = "./models/sentencepiece"
CORPUS_PATH = "./data/KNBC_v1.0_090925/corpus2/"

def get_path_data(corpus_path):
    files = os.listdir(corpus_path)
    keys = [f.split(".")[0] for f in files]
    files_full_path = glob.glob(corpus_path + "/*")
    return [(k, v) for k, v in zip(keys, files_full_path)]

def load_data(corpus_path):
    x = []
    t = []
    for k, path in get_path_data(corpus_path):
        df = pd.read_csv(path, encoding="EUC-JP", delimiter="\t", header=None)
        x.extend([i for i in df[1].tolist()])
        t.extend([k for _ in df[1].tolist()])
    train_x, test_x, train_t, test_t = train_test_split(x, t, test_size=RATE, random_state=SEED)
    return to_df(train_x, train_t), to_df(test_x, test_t)

def to_df(x, t):
    df = pd.DataFrame([(tt, xx) for xx, tt in zip(x, t)])
    df.columns = ["label", "text"]
    return df

def shuffle_list(l):
    rand_i = random.sample(range(len(l)), len(l))
    return [l[i] for i in rand_i]

def parse_batch(batch):
    x = [x for _, x in batch]
    t = np.array([t for t, _ in batch], dtype=np.int32)
    return x, t

def generate_bath(data, size):
    data = shuffle_list(data)
    batch = []
    for d in data:
        batch.append(d)
        if len(batch) >= size:
            yield parse_batch(batch)
            batch = []

def set_seed():
    np.random.seed(SEED)
    random.seed(SEED)

def get_train_data(pad, mode):
    # load data
    train, test = load_data(CORPUS_PATH)
    # init separator
    sep = Separator(mode, SENTENCEPIECE_PATH)
    sep.train(train)
    sentencepiece = mode == "sentencepiece"
    # split text
    train = sep.sep_df_text(train, sentencepiece)
    test = sep.sep_df_text(test, sentencepiece)
    # word to id
    str2idx = Str2idx(train)
    n_vocab = str2idx.get_n_vocab()
    train = str2idx(train, pad)
    test = str2idx(test, pad)
    return train, test, n_vocab
