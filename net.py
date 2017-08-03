# -*- coding: utf-8 -*-

import numpy as np
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L


class MLP(Chain):

    def __init__(self, n_vocab, n_units, n_layers, n_out, use_dropout=0.5, ignore_label=-1):
        super(MLP, self).__init__()
        with self.init_scope():
            self.word_embed=L.EmbedID(n_vocab, n_units, ignore_label=-1)
            self.bi_lstm=L.NStepBiLSTM(n_layers=n_layers, in_size=n_units,
                                    out_size=n_units, dropout=use_dropout)
        self.use_dropout = use_dropout

    def forward(self, x_list):
        """
        Args:
            - x: word index list(np.int32)
        """
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
