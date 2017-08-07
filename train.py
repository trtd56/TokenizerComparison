# -*- cording: utf-8 -*-

import argparse
import chainer.functions as F
from chainer import optimizer, optimizers

from net import CNN, LSTM
from log_tracer import LogTracer
from func import get_train_data, set_seed, generate_bath, parse_batch

N_OUT = 4

parser = argparse.ArgumentParser(description='tokenizer comparison')
parser.add_argument("--net", type=str,
                    help="neural network type(lstm / cnn)")
parser.add_argument("--tokenizer", type=str,
                    help="separate type(character / mecab / sentencepiece / neologd)")
parser.add_argument("--n_units", type=int, default=256,
                    help="number of unit")
parser.add_argument("--n_batch", type=int, default=32,
                    help="number of mini batch")
parser.add_argument("--n_epoch", type=int, default=10,
                    help="number of epoch")

args = parser.parse_args()
nn_type = args.net
sep_mode = args.tokenizer
n_units = args.n_units
n_batch = args.n_batch
n_epoch = args.n_epoch
pad = nn_type == "cnn"

if __name__ == "__main__":
    set_seed()
    log_tracer = LogTracer(nn_type, sep_mode)

    log_tracer("get train data")
    train, test, n_vocab = get_train_data(pad, sep_mode)

    if nn_type == "lstm":
        mlp = LSTM(n_vocab, n_units, N_OUT)
    elif nn_type == "cnn":
        mlp = CNN(n_vocab, n_units, N_OUT)
    opt = optimizers.Adam()
    opt.setup(mlp)

    log_tracer("start train")
    for epoch in range(n_epoch):
        for x, t in generate_bath(train, n_batch):
            mlp.cleargrads()
            loss, acc = mlp(x, t, train=True)
            loss.backward()
            opt.update()
            log_tracer.trace_train(epoch, loss.data, acc.data)
        x_v, t_v = parse_batch(test)
        loss_v, acc_v = mlp(x_v, t_v)
        log_tracer.trace_test(epoch, loss_v.data, acc_v.data, True)
    mlp.save(sep_mode)
