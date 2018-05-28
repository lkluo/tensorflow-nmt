# coding=utf-8

"""Copy from https://github.com/JayParks/tf-seq2seq, with minimum modification"""

import json
import numpy as np
from collections import Counter
import os

# maximum input and output length
# MAX_SOURCE_LEN = 10
# MAX_TARGET_LEN = 10


# Extra vocabulary symbols
_GO = '<s>'
_EOS = '</s>' # also function as PAD
_UNK = 'unk'
_UN = '<u>'
EXTRA_TOKENS = [_GO, _EOS, _UNK, _UN]

START_TOKEN = EXTRA_TOKENS.index(_GO)	# start_token = 0
END_TOKEN = EXTRA_TOKENS.index(_EOS)	# end_token = 1
UNK_TOKEN = EXTRA_TOKENS.index(_UNK)


def build_vocab(filename, max_vocab_size=None):
    word_counter = Counter()
    vocab = dict()
    with open(filename, 'r') as f:
        for line in f:
            words_in = line.strip().split(' ')
            word_counter.update(words_in)
    if max_vocab_size == None or max_vocab_size > len(word_counter):
        max_vocab_size = len(word_counter)

    vocab[_GO] = 0
    vocab[_EOS] = 1
    vocab[_UNK] = 2
    vocab_idx = len(vocab)
    for key, value in word_counter.most_common(max_vocab_size):
        vocab[key] = vocab_idx
        vocab_idx += 1

    with open('%s.json' % filename, 'w') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    # return vocab
    print('Vocab of size {} is built'.format(max_vocab_size))

def load_dict(filename):
    with open(filename, 'r', encoding='utf-8') as json_file:  # add utf-8 for window
        vocab = json.load(json_file)
    return vocab

def load_reverse_dict(filename):
    reverse_vocab = dict()
    vocab = load_dict(filename)
    for key, value in vocab.items():
        reverse_vocab[value] = key
    return reverse_vocab

def token2ids(word, vocab):
    # unknown token
    if word not in vocab:
        word = _UNK
    return vocab[word]

def ids2token(ids,reverse_vocab):
    if ids not in reverse_vocab:
        return _UNK
    else:
        return reverse_vocab[ids]

def sentence2ids(sent, vocab, max_sentence_length, mode='source'):
    """
    sentence to index, with paddings
    :param sent: 
    :param vocab: 
    :param max_sentence_length: 
    :param mode: 
    :return: 
    """
    tokens = sent.split(' ') # use space to tokenize
    sent_len = len(tokens)
    pad_len = max_sentence_length - sent_len
    # return sentence indexes and length
    if mode == 'source':
        return [token2ids(token, vocab) for token in tokens] + [END_TOKEN] * pad_len, sent_len
    else:
        return [START_TOKEN] + [token2ids(token, vocab) for token in tokens] + [END_TOKEN] * pad_len, sent_len + 1

def ids2sentence(indices, reverse_vocab):
    words = []
    for id in indices:
        if id == END_TOKEN:
            break
        word = ids2token(id, reverse_vocab)
        if word != _GO:
            words.append(word)
    return ' '.join(words)
    # return ' '.join([ids2token(ids, reverse_vocab) for ids in indices])

# batch preparation of a given sequence
def prepare_batch(seqs_x, maxlen=None):
    # seqs_x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    batch_size = len(seqs_x)

    x_lengths = np.array(lengths_x)
    maxlen_x = np.max(x_lengths)

    x = np.ones((batch_size, maxlen_x)).astype('int32') * END_TOKEN

    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
    return x, x_lengths


# batch preparation of a given sequence pair for training
def prepare_train_batch(seqs_x, seqs_y, maxlen=None):
    # seqs_x, seqs_y: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x <= maxlen and l_y <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    batch_size = len(seqs_x)

    x_lengths = np.array(lengths_x)
    y_lengths = np.array(lengths_y)

    maxlen_x = np.max(x_lengths)
    maxlen_y = np.max(y_lengths)

    x = np.ones((batch_size, maxlen_x)).astype('int32') * END_TOKEN
    y = np.ones((batch_size, maxlen_y)).astype('int32') * END_TOKEN

    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[idx, :lengths_x[idx]] = s_x
        y[idx, :lengths_y[idx]] = s_y
    return x, x_lengths, y, y_lengths

def create_if_need(dir):
    if not os.path.exists(dir):
        print('Creating path {}'.format(dir))
        os.mkdir(dir)
def remove_if_need(dir):
    if os.path.exists(dir):
        print('Removing path {}'.format(dir))
        os.rmdir(dir)
