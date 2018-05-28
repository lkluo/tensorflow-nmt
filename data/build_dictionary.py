#!/usr/bin/env python

import numpy
import json

import sys

from collections import OrderedDict
from data_utils import EXTRA_TOKENS as extra_tokens

def main():
    for filename in sys.argv[1:]:
        print('Processing', filename)
        word_freqs = OrderedDict()
        with open(filename, 'r') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
        # words = word_freqs.keys()
        # freqs = word_freqs.values()

        # sorted_idx = numpy.argsort(freqs)
        # sorted_words = [words[ii] for ii in sorted_idx[::-1]]
        sorted_words = [key for key, value in word_freqs.items()]

        worddict = OrderedDict()
        for ii, ww in enumerate(extra_tokens):
            worddict[ww] = ii
        for ii, ww in enumerate(sorted_words):
            worddict[ww] = ii + len(extra_tokens)

        with open('%s.json'%filename, 'w') as f:
            json.dump(worddict, f, indent=2, ensure_ascii=False)

        print('{} build dictionary done'.format(filename))

if __name__ == '__main__':
    main()
