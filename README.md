# tensorflow-nmt
A Tensorflow implementation of Neural Machine Translation, based mostly on https://github.com/JayParks/tf-seq2seq

# Example of Chinese-to-English translation
1. download news parallel corpus from WMT2018, e.g.
```data/download.sh```
2. data preprocess, including tokenization (for Chinese sentences, it would be better to conduct sengmentation first, e.g., using jieba), lowercasing, byte-pair-encoding. You may use
```
bash data/preprocess.sh
```
