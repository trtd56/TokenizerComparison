# -*- cording: utf-8 -*-

from chainer import optimizer, optimizers
import chainer.links as L
import chainer.functions as F
import random
import numpy as np

from net import MLP
from str2idx import Str2idx
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
            yield batch
            batch = []

def parse_batch(batch):
    x = [x for _, x in batch]
    t = np.array([t for t, _ in batch], dtype=np.int32)
    return x, t

train, test = load_data("./data/KNBC_v1.0_090925/corpus2/")


train["text"] = train["text"].apply(wakati_mecab)
test["text"] = test["text"].apply(wakati_mecab)


str2idx = Str2idx(train)
train = str2idx(train)
test = str2idx(test)


n_vocab = 5000
n_units = 50
n_layers = 1
n_batch = 50
n_out = 4
n_epoch = 10

mlp = MLP(n_vocab, n_units, n_layers, n_out)

opt = optimizers.SGD()
opt.setup(mlp)

for epoch in range(n_epoch):
    for b in generate_bath(train, n_batch):
        x, t = parse_batch(b)
        mlp.cleargrads()
        loss = mlp(x, t)
        loss.backward()
        opt.update()
        loss.unchain_backward()

        x, t = parse_batch(test)
        y = mlp(x)
        acc = F.accuracy(y, t)
        print("{}\t{}\t{}".format(epoch, loss.data, acc.data))
