# -*- cording: utf-8 -*-

from datetime import datetime
from collections import defaultdict

class LogTracer():

    def __init__(self, nn_type, sep_mode, path="./data/log/"):
        now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.path_t = path + "/" + "train_log_" + now + ".csv"
        log = "{},{},{},{}".format("time", "epoch", "loss", "acc")
        with open(self.path_t, "w") as f:
            f.write(log)
        self.path_v = path + "/" + "test_log_" + now + ".csv"
        log = "{},{},{},{},{},{}".format("time", "epoch", "loss_avg", "acc_avg", "loss_v", "acc_v")
        with open(self.path_v, "w") as f:
            f.write(log)
        self.train_loss = []
        self.train_acc = []

    def trace_train(self, epoch, loss, acc):
        now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        log = "\n{},{},{},{}".format(now, epoch, loss, acc)
        with open(self.path_t, "a") as f:
            f.write(log)
        self.train_loss.append(loss)
        self.train_acc.append(acc)

    def trace_test(self, epoch, loss, acc, trace=False):
        now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        loss_t = sum(self.train_loss)/len(self.train_loss)
        acc_t = sum(self.train_acc)/len(self.train_acc)
        log = "\n{},{},{},{},{},{}".format(now, epoch, loss_t, acc_t, loss, acc)
        with open(self.path_v, "a") as f:
            f.write(log)
        if trace:
            print("{}\t{}\t{}\t{}\t{}\t{}".format(now, epoch, loss_t, acc_t, loss, acc))
        self.train_loss = []
        self.train_acc = []

    def trace_label(self, d_type, data):
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        l_freq = defaultdict(int)
        for i in data:
            l_freq[int(i[0])] += 1
        hyp_max_acc = max(l_freq.values())/sum(l_freq.values())
        log = "{}\t{}\t{}\t{}".format(now, d_type, "\t".join(["{}: {}".format(k, v)for k, v in l_freq.items()]), hyp_max_acc)
        print(log)

    def __call__(self, out):
        now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        print("{}\t{}".format(now, out))
