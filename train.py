# -*- cording: utf-8 -*-

import random
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import optimizer, optimizers

from net import MLP
from str2idx import Str2idx
from log_tracer import LogTracer
from make_train_data import load_data, wakati_mecab

def shuffle_list(l):
    rand_i = random.sample(range(len(l)), len(l))
    return [l[i] for i in rand_i]

def generate_bath(data, size):
    data = shuffle_list(data)
    batch = []
    for d in data:
        batch.append(d)
        if len(batch) >= size:
            yield parse_batch(batch)
            batch = []

def parse_batch(batch):
    x = [x for _, x in batch]
    t = np.array([t for t, _ in batch], dtype=np.int32)
    return x, t

def get_train_data(path, wakati_func):
    train, test = load_data(path)
    train["text"] = train["text"].apply(wakati_func)
    test["text"] = test["text"].apply(wakati_func)
    str2idx = Str2idx(train)
    train = str2idx(train)
    test = str2idx(test)
    return train, test

corpus_path = "./data/KNBC_v1.0_090925/corpus2/"
n_vocab = 5000
n_units = 50
n_layers = 1
n_batch = 50
n_out = 4
n_epoch = 10

train, test = get_train_data(corpus_path, wakati_mecab)
log_tracer = LogTracer()

mlp = MLP(n_vocab, n_units, n_layers, n_out)
opt = optimizers.SGD()
opt.setup(mlp)

for epoch in range(n_epoch):
    for x, t in generate_bath(train, n_batch):
        mlp.cleargrads()
        loss = mlp(x, t)
        loss.backward()
        opt.update()
        loss.unchain_backward()

        x, t = parse_batch(test)
        y = mlp(x)
        acc = F.accuracy(y, t)
        log_tracer(epoch, loss.data, acc.data)
