
import sys
import numpy as np
import json

def main(argv):
    sentence_length = []
    vocab_size = []
    for input_file in argv:
        lengths = []
        with open(input_file, 'r') as corpus:
            for line in corpus:
                lengths.append(len(line.split()))
            print("%s: size=%d, avg_length=%.2f, std=%.2f, min=%d, max=%d" 
              % (input_file, len(lengths), np.mean(lengths), np.std(lengths), np.min(lengths), np.max(lengths)))
            sentence_length.append(len(lengths))
        # dictionary
        with open(input_file+'.json') as f:
            vocab = json.load(f)
            print("%s: dictionary size=%d"
                  % (input_file+'.json', len(vocab)))
            vocab_size.append(len(vocab))

    # write statistic
    with open(input_file+'.txt', 'w') as f:
        f.write('sentences lengths: {}, {}'.format(sentence_length[0], sentence_length[1]))
        f.write('\n')
        f.write('vocab size: {}, {}'.format(vocab_size[0], vocab_size[1]))


if __name__ == "__main__":
    main(sys.argv[1:])
