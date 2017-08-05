# -*- coding: utf-8 -*-

import numpy as np
from chainer import Chain, ChainList, Variable
import chainer.functions as F
import chainer.links as L


class LSTM(Chain):

    def __init__(self, n_vocab, n_units, n_layers, n_out, use_dropout=0.5, ignore_label=-1):
        super(LSTM, self).__init__()
        with self.init_scope():
            self.word_embed=L.EmbedID(n_vocab, n_units, ignore_label=-1)
            self.bi_lstm=L.NStepBiLSTM(n_layers=n_layers, in_size=n_units,
                                    out_size=n_units, dropout=use_dropout)
        self.use_dropout = use_dropout

    def forward(self, x_list):
        xs = []
        hx = None 
        cx = None
        for x in x_list:
            x = Variable(x)
            x = self.word_embed(x)
            x = F.dropout(x, ratio=self.use_dropout)
            xs.append(x)
        hy, cy, ys = self.bi_lstm(hx=hx, cx=cx, xs=xs)
        y = F.concat(hy, axis=1)

        return y

    def __call__(self, x_list, t=None):
        y = self.forward(x_list)
        if t is None:
            return y
        else:
            return F.softmax_cross_entropy(y, t)

class CNN(ChainList):

    def __init__(self, n_vocab, n_units, n_layers, n_out, ksize, stride, use_dropout=0.5, ignore_label=-1):
        super(CNN, self).__init__()
        with self.init_scope():
            self.word_embed=L.EmbedID(n_vocab, n_units, ignore_label=-1)
            self.conv1 = L.Convolution2D(None, n_units, (ksize, n_units), stride, pad=(ksize, 0))
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_out)
        self.use_dropout = use_dropout

    def forward(self, x):
        x = Variable(x)
        x = self.word_embed(x)
        x = F.dropout(x, ratio=self.use_dropout)
        x = F.expand_dims(x, axis=1)
        x = F.relu(self.conv1(x))
        x = F.max_pooling_2d(x, 3)
        x = F.dropout(F.relu(self.l1(x)), ratio=self.use_dropout)
        x = self.l2(x)
        return x

    def __call__(self, x_list, t=None):
        x = self.xp.array(x_list, dtype=self.xp.int32)
        y = self.forward(x)
        if t is None:
            return y
        else:
            return F.softmax_cross_entropy(y, t)
