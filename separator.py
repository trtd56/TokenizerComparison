# -*- coding: utf-8 -*-

import os
import MeCab
import subprocess
import pandas as pd

class Separator():
    __NATIVE_TEXT = "native.txt"
    __MODEL_PATH = "sentencepiece"
    __BULK_DATA = "bulk_data.txt"

    def __init__(self, mode, path):
        if mode == "mecab":
            self.wakati = MeCab.Tagger("-Ochasen")
        elif mode == "neologd":
            self.wakati = MeCab.Tagger(" -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
        self.mode = mode
        self.path = path
        self.native_text = path + "/" + self.__NATIVE_TEXT
        self.model_path = path + "/" + self.__MODEL_PATH

    def wakati_mecab(self, doc):
        doc = WAKATI.parse(doc)
        doc = [i.split("\t")[0] for i in doc.split("\n")]
        doc = [d for d in doc if not d in ["", "EOS"]]
        return doc

    def train_sentencepiece(self, vocab_size):
        cmd = "spm_train --input=" + self.native_text + \
                " --model_prefix=" + self.model_path + \
                " --vocab_size=" + str(vocab_size)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_data, stderr_data = p.communicate()
        with open(self.path + "/stdout.log", "w") as f:
            f.write(stdout_data.decode("utf-8"))
        with open(self.path + "/stderr.log", "w") as f:
            f.write(stderr_data.decode("utf-8"))


    def train(self, train_data, vocab_size=8000):
        if self.mode == "sentencepiece":
            train_data["text"].to_csv(self.native_text, index=None)
            self.train_sentencepiece(vocab_size)

    def wakati_sentencepiece(self, doc):
        cmd = "echo " + str(doc) + " | spm_encode --model=" + \
                self.model_path + " --output_format=piece"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout_data, stderr_data = p.communicate()
        return stdout_data.decode("utf-8").split()

    def __call__(self, doc):
        if self.mode in ["mecab", "neologd"]:
            doc = self.wakati_mecab(doc)
        elif self.mode == "sentencepiece":
            doc = self.wakati_sentencepiece(doc)
        elif self.mode == "character":
            doc = ["_" if i == " " else i for i in doc]
        return " ".join(doc)

    def sep_bulk(self, text_list):
        bulk = self.path + "/" + self.__BULK_DATA
        try:
            with open(bulk, "w") as f:
                f.write("\n".join(text_list))
            cmd = "spm_encode --model=" + self.model_path + ".model --output_format=piece < " + bulk
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout_data, stderr_data = p.communicate()
        finally:
            os.remove(bulk)
        return stdout_data.decode("utf-8").split("\n")

    def sep_df_text(self, df, bulk=False):
        if bulk:
            df["text"] = pd.DataFrame(self.sep_bulk(df["text"].tolist()))
        else:
            df["text"] = df["text"].apply(self)
        return df
