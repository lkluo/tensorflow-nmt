# tensorflow-nmt
A Tensorflow implementation of Neural Machine Translation, based mostly on https://github.com/JayParks/tf-seq2seq. 
Support Tensorflow >= 1.2.1 and GPU.

Features:
* Bidirectional LSTM
* Learning rate decay

## Example of Chinese-to-English translation
1. download news parallel corpus from WMT2018, e.g.

```
bash data/download.sh
```
2. data preprocess, including tokenization (for Chinese sentences, it would be better to conduct sengmentation first, e.g., using jieba), lowercasing, byte-pair-encoding. You may use

```
bash data/preprocess.sh
```

3. model training
```
python3 -m train \
--model_dir=model-zh-en/ \
--embedding_size=512 \
--hidden_units=512 \
--batch_size=128 \
--start_decay_step=100000 \
--decay_steps=30000 \
--display_freq=80 \
--save_freq=10000 \
--source_vocabulary=data/zh-en/news-commentary-v13.zh-en.final.zh.json \
--target_vocabulary=data/zh-en/news-commentary-v13.zh-en.final.en.json \
--source_train_data=data/zh-en/news-commentary-v13.zh-en.final.zh \
--target_train_data=data/zh-en/news-commentary-v13.zh-en.final.en
```
or just
```
bash train.sh
```
