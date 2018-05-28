# coding=utf-8

"""Copy from https://github.com/JayParks/tf-seq2seq, with minimum modification"""

import os
import math
import time
import json
import csv

import numpy as np
import tensorflow as tf

from data_iterator import BiTextIterator

import data_utils as data_utils
from data_utils import prepare_train_batch

from seq2seq_model import Seq2SeqModel

import argparse

"""Build ArgumentParser"""
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

# data
parser.add_argument('--source_vocabulary', type=str, default='', help='Source vocab path')
parser.add_argument('--target_vocabulary', type=str, default='', help='Target vocab path')
parser.add_argument('--source_train_data', type=str, default='', help='Train source path')
parser.add_argument('--target_train_data', type=str, default='', help='Train source path')
parser.add_argument('--source_valid_data', type=str, default='', help='Dev source path')
parser.add_argument('--target_valid_data', type=str, default='', help='Dev target path')

# Network parameters
parser.add_argument('--cell_type', type=str, default='lstm', help='lstm|gru')
parser.add_argument('--attention_type', type=str, default='bahdanau', help='bahdanau|luong')
parser.add_argument('--hidden_units', type=int, default=512, help='Network size')
parser.add_argument('--embedding_size', type=int, default=512, help='Embedding size')
parser.add_argument('--depth', type=int, default=4, help='Network depth')
parser.add_argument('--num_encoder_symbols', type=int, default=-1, help='Source vocabulary size')
parser.add_argument('--num_decoder_symbols', type=int, default=-1, help='Target vocabulary size')
parser.add_argument('--bidirectional', type=str2bool, nargs="?", const=True, default=True, help='Bidirectional cell: True|False')

parser.add_argument('--use_residual', type=str2bool, nargs="?", const=True, default=True, help='Residual connection between layers: True|False')
parser.add_argument('--attn_input_feeding', type=str2bool, nargs="?", const=False, default=False, help='Input feeding method in attention decoder: True|False')
parser.add_argument('--use_dropout', type=str2bool, nargs="?", const=True, default=True, help='Dropout: True|False')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate: 1-keep_prob')

# Optimizer
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer for training: adadelta|adam|rmsprop|sgd')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learing rate')  # <0.001 if using adam
parser.add_argument('--start_decay_step', type=int, default=9, help='When to start to decay')
parser.add_argument('--decay_steps', type=int, default=1, help='How frequent to decay')
parser.add_argument('--decay_factor', type=float, default=0.5, help='How much to decay')
parser.add_argument('--max_gradient_norm', type=float, default=5.0, help='Clip gradients to this norm')

# Load data
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--max_epochs', type=int, default=30, help='Maximum epochs')
parser.add_argument('--max_load_batches', type=int, default=20, help='Maximum # of batches to load at one time')
parser.add_argument('--max_seq_length', type=int, default=80, help='Maximum sequence length')
parser.add_argument('--shuffle_each_epoch', type=str2bool, nargs="?", const=True, default=True,  help='Shuffle training dataset for each epoch')
parser.add_argument('--sort_by_length', type=str2bool, nargs="?", const=True, default=True, help='Sort pre-fetched minibatches by their target sequence lengths')

# Display when training
parser.add_argument('--display_freq', type=int, default=100, help='Display training status every this iteration')
parser.add_argument('--save_freq', type=int, default=1, help='Save model checkpoint every this iteration')
parser.add_argument('--valid_freq', type=int, default=1, help='Evaluate model every this iteration: valid_data needed')
parser.add_argument('--model_dir', type=str, default='model/', help='Path to save model checkpoints')
parser.add_argument('--model_name', type=str, default='translate.ckpt', help='Path to save model checkpoints')

# Runtime parameters
parser.add_argument('--allow_soft_placement', type=str2bool, nargs="?", const=True, default=True, help='Allow device soft placement')
parser.add_argument('--log_device_placement', type=str2bool, nargs="?", const=False, default=False, help='Log placement of ops on devices')

# Other
parser.add_argument('--use_fp16', type=str2bool, nargs="?", const=False, default=False, help='Use half precision float16 instead of float32 as dtype: True|False')
# parser.add_argument('--self_decay', type=bool, default=True, help='Use self calculate settings for SGD optimization')

FLAGS = parser.parse_args()

def check():
  if tf.__version__ < "1.2.1":
    raise EnvironmentError("Tensorflow version must >= 1.2.1")

  if FLAGS.bidirectional and FLAGS.cell_type == "gru":
      raise NotImplementedError("Support only bidirectional LSTM")

def create_model(session, FLAGS):

    #config = OrderedDict(sorted(FLAGS.__flags.items()))
    model = Seq2SeqModel(FLAGS, 'train')

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.restore(session, ckpt.model_checkpoint_path)
        
    else:
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        print('Creating new model parameters..')
        session.run(tf.global_variables_initializer())
   
    return model

def load_data(FLAGS):
    # Load parallel data to train
    print('Loading training data..')
    train_set = BiTextIterator(source=FLAGS.source_train_data,
                               target=FLAGS.target_train_data,
                               source_dict=FLAGS.source_vocabulary,
                               target_dict=FLAGS.target_vocabulary,
                               batch_size=FLAGS.batch_size,
                               maxlen=FLAGS.max_seq_length,
                               n_words_source=FLAGS.num_encoder_symbols,
                               n_words_target=FLAGS.num_decoder_symbols,
                               shuffle_each_epoch=FLAGS.shuffle_each_epoch,
                               sort_by_length=FLAGS.sort_by_length,
                               maxibatch_size=FLAGS.max_load_batches)

    if FLAGS.source_valid_data and FLAGS.target_valid_data:
        print('Loading validation data..')
        valid_set = BiTextIterator(source=FLAGS.source_valid_data,
                                   target=FLAGS.target_valid_data,
                                   source_dict=FLAGS.source_vocabulary,
                                   target_dict=FLAGS.target_vocabulary,
                                   batch_size=FLAGS.batch_size,
                                   maxlen=None,
                                   n_words_source=FLAGS.num_encoder_symbols,
                                   n_words_target=FLAGS.num_decoder_symbols)
    else:
        valid_set = None

    return train_set, valid_set

def train():
    # check tensorflow version
    check()

    data_utils.create_if_need(FLAGS.model_dir)
    # Load train and valid data
    train_set, valid_set = load_data(FLAGS)

    # calculate training information
    source_dict_size = len(data_utils.load_dict(FLAGS.source_vocabulary))
    target_dict_size = len(data_utils.load_dict(FLAGS.target_vocabulary))
    print('source dict size: {}\ntarget dict size: {}\n'.format(source_dict_size, target_dict_size))

    if FLAGS.num_encoder_symbols < 0:
        FLAGS.num_encoder_symbols = source_dict_size + 1
    if FLAGS.num_decoder_symbols < 0:
        FLAGS.num_decoder_symbols = target_dict_size + 1


    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, 
        log_device_placement=FLAGS.log_device_placement, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

        # Create a log writer object
        log_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

        # Create a new model or reload existing checkpoint
        model = create_model(sess, FLAGS)

        step_time, loss = 0.0, 0.0
        words_seen, sents_seen = 0, 0
        start_time = time.time()

        # Training loop
        print('Training..')
        for epoch_idx in range(FLAGS.max_epochs):

            if model.global_epoch_step.eval() >= FLAGS.max_epochs:
                print('Training is already complete.', \
                      'current epoch:{}, max epoch:{}'.format(model.global_epoch_step.eval(), FLAGS.max_epochs))
                break

            for source_seq, target_seq in train_set:
                # Get a batch from training parallel data
                source, source_len, target, target_len = prepare_train_batch(source_seq, target_seq,
                                                                             FLAGS.max_seq_length)
                if source is None or target is None:
                    print('No samples under max_seq_length ', FLAGS.max_seq_length)
                    continue

                # Execute a single training step
                step_loss, summary = model.train(sess, encoder_inputs=source, encoder_inputs_length=source_len, 
                                                 decoder_inputs=target, decoder_inputs_length=target_len)

                loss += float(step_loss) / FLAGS.display_freq
                words_seen += float(np.sum(source_len+target_len))
                sents_seen += float(source.shape[0]) # batch_size

                if model.global_step.eval() % FLAGS.display_freq == 0:

                    avg_perplexity = math.exp(float(loss)) if loss < 300 else float("inf")

                    time_elapsed = time.time() - start_time
                    step_time = time_elapsed / FLAGS.display_freq

                    words_per_sec = words_seen / time_elapsed
                    sents_per_sec = sents_seen / time_elapsed

                    print('Epoch ', model.global_epoch_step.eval() + 1, 'Step ', model.global_step.eval(), \
                          'Perplexity {0:.2f}'.format(avg_perplexity), 'Step-time {0:.2f}'.format(step_time), \
                          '{0:.2f} sents/s'.format(sents_per_sec), '{0:.2f} words/s'.format(words_per_sec))

                    loss = 0
                    words_seen = 0
                    sents_seen = 0
                    start_time = time.time()

                    # Record training summary for the current batch
                    log_writer.add_summary(summary, model.global_step.eval())

                # Execute a validation step
                if valid_set and model.global_step.eval() % FLAGS.valid_freq == 0:
                    print('Validation step')
                    valid_loss = 0.0
                    valid_sents_seen = 0
                    for source_seq, target_seq in valid_set:
                        # Get a batch from validation parallel data
                        source, source_len, target, target_len = prepare_train_batch(source_seq, target_seq)

                        # Compute validation loss: average per word cross entropy loss
                        step_loss, summary = model.eval(sess, encoder_inputs=source, encoder_inputs_length=source_len,
                                                        decoder_inputs=target, decoder_inputs_length=target_len)
                        batch_size = source.shape[0]

                        valid_loss += step_loss * batch_size
                        valid_sents_seen += batch_size
                        print('  {} samples seen'.format(valid_sents_seen))

                    valid_loss = valid_loss / valid_sents_seen
                    print('Valid perplexity: {0:.2f}'.format(math.exp(valid_loss)))

                # Save the model checkpoint
                if model.global_step.eval() % FLAGS.save_freq == 0:
                    print('Saving the model..')
                    checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                    model.save(sess, checkpoint_path, global_step=model.global_step)
                    json.dump(vars(model.config),
                              open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'w', encoding='utf-8'),
                              indent=2)

                    # Save loss
                    try:
                        with open(os.path.join(FLAGS.model_dir, 'loss') + '.csv', 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([model.global_step.eval(), avg_perplexity, math.exp(valid_loss)])
                            print('Loss saved ..')
                    except:
                        pass

            # Increase the epoch index of the model
            model.global_epoch_step_op.eval()
            print('Epoch {0:} DONE'.format(model.global_epoch_step.eval()))
        
        print('Saving the last model..')
        checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
        model.save(sess, checkpoint_path, global_step=model.global_step)
        json.dump(vars(model.config),
                  open('%s-%d.json' % (checkpoint_path, model.global_step.eval()), 'w', encoding='utf-8'),
                  indent=2)
        
    print('Training Terminated')



def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
