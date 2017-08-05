# -*- cording: utf-8 -*-

import chainer.functions as F
from chainer import optimizer, optimizers

import net
from log_tracer import LogTracer
from func import get_train_data, set_seed, generate_bath, parse_batch

nn_type = "cnn"
sep_mode = "character"
n_vocab = 5000
n_units = 50
n_layers = 1
n_batch = 50
n_out = 4
n_epoch = 10
ksize = 5
stride = 1


if __name__ == "__main__":
    set_seed()
    log_tracer = LogTracer()

    log_tracer.trace("set network")
    if nn_type == "lstm":
        mlp = net.LSTM(n_vocab, n_units, n_layers, n_out)
    elif nn_type == "cnn":
        mlp = net.CNN(n_vocab, n_units, n_layers, n_out, ksize, stride)
    opt = optimizers.SGD()
    opt.setup(mlp)

    log_tracer.trace("get train data")
    pad = type(mlp) is net.CNN
    train, test = get_train_data(pad, sep_mode)

    log_tracer.trace("start train")
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
            log_tracer(epoch, loss.data, acc.data, trace=True)
