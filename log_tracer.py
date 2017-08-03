# -*- cording: utf-8 -*-

from datetime import datetime

class LogTracer():

    def __init__(self, path="./data/log/"):
        now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.csv")
        self.path = path + "/" + now + "_train_log.csv"
        log = "{}.{},{},{}".format("time", "epoch", "loss", "acc")
        with open(self.path, "w") as f:
            f.write(log)

    def __call__(self, epoch, loss, acc):
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        log = "\n{},{},{},{}".format(now, epoch, loss, acc)
        with open(self.path, "a") as f:
            f.write(log)
