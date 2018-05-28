#!/bin/bash

# source language suffix (example: en, cs, de, fr)
S=zh

# target language suffix (example: en, cs, de, fr)
T=en

# path to corpus
CORPUS=zh-en/news-commentary-v13.zh-en

echo "tokenizing.."
perl tokenizer.perl -l ${S} -threads 10 < ${CORPUS}.${S} > ${CORPUS}.tok.${S}
perl tokenizer.perl -l ${T} -threads 10 < ${CORPUS}.${T} > ${CORPUS}.tok.${T}

echo "lowercasing.."
perl lowercase.perl < ${CORPUS}.tok.${S} > ${CORPUS}.lower.${S}
perl lowercase.perl < ${CORPUS}.tok.${T} > ${CORPUS}.lower.${T}

echo "learning bpe.."
# learn BPE on joint vocabulary
cat ${CORPUS}.lower.${S} ${CORPUS}.lower.${T} | python subword_nmt/learn_bpe.py -s 30000 > ${S}${T}.bpe

echo "applying bpe.."
python3 subword_nmt/apply_bpe.py -c ${S}${T}.bpe < ${CORPUS}.tok.${S} > ${CORPUS}.bpe.${S}
python3 subword_nmt/apply_bpe.py -c ${S}${T}.bpe < ${CORPUS}.tok.${T} > ${CORPUS}.bpe.${T}

echo "shuffling.."
python3 shuffle.py ${CORPUS}.bpe.${S} ${CORPUS}.bpe.${T}


mv ${CORPUS}.bpe.${S}.shuf ${CORPUS}.final.${S}
mv ${CORPUS}.bpe.${T}.shuf ${CORPUS}.final.${T}

echo "building dictionaries.."
python3 build_dictionary.py ${CORPUS}.final.${S} ${CORPUS}.final.${T}
