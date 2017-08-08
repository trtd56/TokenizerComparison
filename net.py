# -*- coding: utf-8 -*-

import numpy as np
from chainer import Chain, Variable, serializers, initializers, using_config
import chainer.functions as F
import chainer.links as L


class LSTM(Chain):

    def __init__(self, n_vocab, n_units,n_out, n_layers=2, use_dropout=0.1, ignore_label=-1):
        super(LSTM, self).__init__()
        with self.init_scope():
            self.word_embed=L.EmbedID(n_vocab, n_units, ignore_label=-1)
            self.bi_lstm=L.NStepBiLSTM(n_layers=n_layers, in_size=n_units,
                                    out_size=n_units, dropout=use_dropout)
        self.use_dropout = use_dropout

    def forward(self, x_list, train):
        with using_config('train', train):
            dropout = self.use_dropout if train else 0
            xs = []
            hx = None 
            cx = None
            for x in x_list:
                x = Variable(x)
                x = self.word_embed(x)
                x = F.dropout(x, ratio=dropout)
                xs.append(x)
            hy, cy, ys = self.bi_lstm(hx=hx, cx=cx, xs=xs)
            y = F.concat(hy, axis=1)
        return y

    def __call__(self, x_list, t=None, train=False):
        y = self.forward(x_list, train)
        if t is None:
            return y
        else:
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def save(self, mode, path="./models/net/"):
        serializers.save_npz(path + mode + "_lstm_model.npz", self)


class CNN(Chain):

    def __init__(self, n_vocab, n_units, n_out, filter_size=(3, 4, 5), stride=1, use_dropout=0.5, ignore_label=-1):
        super(CNN, self).__init__()
        initializer = initializers.HeNormal()
        with self.init_scope():
            self.word_embed=L.EmbedID(n_vocab, n_units, ignore_label=-1)
            self.conv1 = L.Convolution2D(None, n_units, (filter_size[0], n_units), stride, pad=(filter_size[0], 0), initialW=initializer)
            self.conv2 = L.Convolution2D(None, n_units, (filter_size[1], n_units), stride, pad=(filter_size[1], 0), initialW=initializer)
            self.conv3 = L.Convolution2D(None, n_units, (filter_size[2], n_units), stride, pad=(filter_size[2], 0), initialW=initializer)
            self.norm1 = L.BatchNormalization(n_units)
            self.norm2 = L.BatchNormalization(n_units)
            self.norm3 = L.BatchNormalization(n_units)
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_out)
        self.use_dropout = use_dropout
        self.filter_size = filter_size

    def forward(self, x, train):
        with using_config('train', train):
            x = Variable(x)
            x = self.word_embed(x)
            x = F.dropout(x, ratio=self.use_dropout)
            x = F.expand_dims(x, axis=1)
            x1 = F.relu(self.norm1(self.conv1(x)))
            x1 = F.max_pooling_2d(x1, self.filter_size[0])
            x2 = F.relu(self.norm2(self.conv2(x)))
            x2 = F.max_pooling_2d(x2, self.filter_size[1])
            x3 = F.relu(self.norm3(self.conv3(x)))
            x3 = F.max_pooling_2d(x3, self.filter_size[2])
            x = F.concat((x1, x2, x3), axis=2)
            x = F.dropout(F.relu(self.l1(x)), ratio=self.use_dropout)
            x = self.l2(x)
        return x

    def __call__(self, x_list, t=None, train=False):
        x = self.xp.array(x_list, dtype=self.xp.int32)
        y = self.forward(x, train)
        if t is None:
            return y
        else:
            return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    def save(self, mode, path="./models/net/"):
        serializers.save_npz(path + mode + "_cnn_model.npz", self)
