# -*- cording: utf-8 -*-

import os
import glob
import MeCab
import pandas as pd
from sklearn.model_selection import train_test_split

WAKATI = MeCab.Tagger("-Ochasen")
SEED = 0
RATE = 0.1

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

def wakati_mecab(doc):
    doc = WAKATI.parse(doc)
    doc = [i.split("\t")[0] for i in doc.split("\n")]
    doc = [d for d in doc if not d in ["", "EOS"]]
    return " ".join(doc)
