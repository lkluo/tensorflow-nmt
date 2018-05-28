# coding: utf-8

"""Copy from https://github.com/JayParks/tf-seq2seq, with minimum modification"""

import json
import tensorflow as tf
from tool.data_iterator import TextIterator
import tool.data_utils as data_utils
from tool.data_utils import prepare_batch
from seq2seq_model import Seq2SeqModel
import glob
import os
import csv

from tool.bleu import _bleu

import argparse
"""Build ArgumentParser"""
parser = argparse.ArgumentParser()

# Decoding parameters
parser.add_argument('--beam_width', type=int, default=10, help='Beam width used in beamsearch')
parser.add_argument('--decode_batch_size', type=int, default=128, help='Batch size used for decoding')
parser.add_argument('--write_n_best', action='store_true', default=False, help='Write n-best list (n=beam_width)')
parser.add_argument('--max_decode_step', type=int, default=500, help='Maximum time step limit to decode')
parser.add_argument('--model_path', type=str, default='model/translate.ckpt-3', help='Path to a specific model checkpoint')
parser.add_argument('--model_checkpoint', nargs='+', default='3', help='Steps to a specific model checkpoint')
parser.add_argument('--decode_input', type=str, default='sample/test.shuf.en', help='Decoding input path')
parser.add_argument('--decode_output', type=str, default='sample/test.de.yue', help='Decoding output path')
parser.add_argument('--decode_reference', type=str, default='sample/test.shuf.yue', help='Decoding reference path')
# parser.add_argument('--decode_multiple', action='store_true', default=False, help='Decoding with multiple models?')
# parser.add_argument('--multiple_path', type=str, default='model/', help='Path to a specific models checkpoint')

# Runtime parameters
# parser.add_argument('--allow_soft_placement', type=str2bool, nargs="?", const=True, default=True, help='Allow device soft placement')
# parser.add_argument('--log_device_placement', type=str2bool, nargs="?", const=False, default=False, help='Log placement of ops on devices')

FLAGS = parser.parse_args()

class DotDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val


def load_config(FLAGS):
    
    config = json.load(open('%s.json' % FLAGS.model_path, 'r'))
    FLAGS = vars(FLAGS) # parser to dictionary
    config.update(FLAGS)

    return DotDict(config)


def load_model(session, config):
    
    model = Seq2SeqModel(config, 'decode')
    if tf.train.checkpoint_exists(config.model_path):
        print('Reloading model parameters..')
        model.restore(session, config.model_path)
    else:
        raise ValueError(
            'No such file:[{}]'.format(config.model_path))
    return model

def load_data(config):
    test_set = TextIterator(source=config.decode_input,
                            batch_size=config.decode_batch_size,
                            source_dict=config.source_vocabulary,
                            maxlen=None,
                            n_words_source=config.num_encoder_symbols)
    # Load inverse dictionary used in decoding
    target_inverse_dict = data_utils.load_reverse_dict(config.target_vocabulary)
    return test_set, target_inverse_dict

def decode_one(config):
    # config = load_config(FLAGS)
    test_set, target_inverse_dict = load_data(config)
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=config.allow_soft_placement,
        log_device_placement=config.log_device_placement, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        # Reload existing checkpoint
        model = load_model(sess, config)

        try:
            print('Decoding {}..'.format(FLAGS.decode_input))
            fout = open(FLAGS.decode_output, 'w', encoding='utf-8')
            for idx, source_seq in enumerate(test_set):
                source, source_len = prepare_batch(source_seq)
                # predicted_ids: GreedyDecoder; [batch_size, max_time_step, 1]
                # BeamSearchDecoder; [batch_size, max_time_step, beam_width]
                predicted_ids = model.predict(sess, encoder_inputs=source,
                                              encoder_inputs_length=source_len)

                # Write decoding results
                for seq in predicted_ids:
                    fout.write(str(data_utils.ids2sentence(seq[:, 0], target_inverse_dict)) + '\n')

                # print('  {}th line decoded'.format(idx * FLAGS.decode_batch_size))

            fout.close()
        except IOError:
            pass

def decode_multiple(FLAGS):
    # model_path = FLAGS.model_path
    model_path, steps = get_steps_per_epoch(FLAGS.model_path)
    for p in FLAGS.model_checkpoint:
        # update config
        step = int(p) * steps
        FLAGS.model_path = model_path + str(step)
        config = load_config(FLAGS)

        decode_one(config)

        # compute bleu (percetage)
        print('Calculating bleu..')
        bleu = _bleu(ref_file=config.decode_reference, trans_file=config.decode_output, subword_option='bpe')
        print('bleu of model {}: {}'.format(p, bleu))

        with open(os.path.join(config.model_dir, 'bleu') + '.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([step, bleu])



def fetch_model(model_dir):
    return glob.glob(os.path.join(model_dir, 'translate.ckpt-*.json'))

def get_steps_per_epoch(model_path):
    ids = model_path.index('-')
    path = model_path[:ids+1]
    return path, int(model_path[ids + 1:])

def decode():
    # Load model config
    config = load_config(FLAGS)
    # Load source data to decode
    test_set, target_inverse_dict = load_data(config)

    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=config.allow_soft_placement,
        log_device_placement=config.log_device_placement, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        # Reload existing checkpoint
        model = load_model(sess, config)
        try:
            print('Decoding {}..'.format(FLAGS.decode_input))
            if FLAGS.write_n_best:
                fout = [open(("%s_%d" % (FLAGS.decode_output, k)), 'w', encoding='utf-8') \
                        for k in range(FLAGS.beam_width)]
            else:
                fout = [open(FLAGS.decode_output, 'w', encoding='utf-8')]

            
            for idx, source_seq in enumerate(test_set):
                source, source_len = prepare_batch(source_seq)
                # predicted_ids: GreedyDecoder; [batch_size, max_time_step, 1]
                # BeamSearchDecoder; [batch_size, max_time_step, beam_width]
                predicted_ids = model.predict(sess, encoder_inputs=source, 
                                              encoder_inputs_length=source_len)
                   
                # Write decoding results
                for k, f in reversed(list(enumerate(fout))):
                    for seq in predicted_ids:
                        f.write(str(data_utils.ids2sentence(seq[:,k], target_inverse_dict)) + '\n')
                    if not FLAGS.write_n_best:
                        break
                print('  {}th line decoded'.format(idx * FLAGS.decode_batch_size))
                
            print('Decoding terminated')
        except IOError:
            pass

# def calbleu():
#     print('Calculate bleu..')
#     candidate, references = fetch_data(FLAGS.decode_output, FLAGS.decode_reference)
#     bleu = BLEU(candidate, references)
#     #with open(config.result_path, 'w') as f:
#     #  f.write(bleu)
#     #print(bleu)
#     return bleu

def main(_):
    decode_multiple(FLAGS)


if __name__ == '__main__':
    tf.app.run()