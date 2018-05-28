#!/bin/bash

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
