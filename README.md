#### 使用データ

[KNPコーパス](http://nlp.ist.i.kyoto-u.ac.jp/index.php?KNP)を使用する.


~~~bash
$ cd ./data
$ wget ftp://ftp.za.freebsd.org/macports/distfiles/KNBC/KNBC_v1.0_090925.tar.bz2
$ tar xf KNBC_v1.0_090925.tar.bz2
~~~
#### 準備

~~~
$ mkdir models/net models/sentencepiece
~~~

#### メモ

SentencePieceをCentOSにインストールする場合は下記モジュールもインストールしておく.

~~~
$ yum install protobuf-devel boost-devel gflags-devel lmdb-devel
~~~
