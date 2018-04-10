import argparse
import glob
import itertools
import logging
import math
import random

import numpy as np
from sklearn import model_selection
import tensorflow as tf
import yaml

import annotate
import data
import vocab


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# TODO:
# -----
# Toggle LSTM direction (forward or backward)
# Annotate with more than just final LSTM output layer
# Clean up session management


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help=('Path to configuration file'))

    parser.add_argument('--train', nargs=1,
                        help='Path to training file')
    parser.add_argument('--validate', nargs='?', default=False,
                        help='Path to validation file. If flag set but with no '
                        'arg, split the training file if one exists')
    parser.add_argument('--test', nargs='?', default=False,
                        help='Path to test file. If flag set but with no arg, '
                             'split the training file if one exists')
    parser.add_argument('--annotate', nargs=2, default=False,
                        help='Paths to error detection files. '
                             'First path is input, second is output.')

    parser.add_argument('--save', help='Path to save checkpoint')
    parser.add_argument('--save_vocab', help='Path to save vocab file')
    parser.add_argument('--load', help='Path to load checkpoint')
    parser.add_argument('--load_vocab', help='Path to load vocab file')
    parser.add_argument('--embeddings', help='Path to load embeddings')

    args = parser.parse_args()

    if (args.test or args.annotate) and not args.train and not (args.load_vocab or args.load):
        parser.error('If not training, you must load a model and vocab')

    if (args.validate is not False and not args.train):
        # Gotcha: args.validate is False if --validate is unset, and None
        # if --validate is set without an argument.
        parser.error('Cannot validate without training')

    if (args.test is None and not args.train):
        # Gotcha: args.test is False if --test is unset, and None if --test is
        # set without an argument.
        parser.error('Must specify a test file if testing without training')

    if args.train and len(glob.glob(args.train[0])) != 1 and args.test is None:
        parser.error('Cannot perform train/test split when streaming')

    if args.train and len(glob.glob(args.train[0])) != 1 and args.validate is None:
        parser.error('Cannot perform train/dev split when streaming')

    return args


def _load_config(path):
    with open(path, 'r') as f:
        config = yaml.load(f)
        print(yaml.dump(config))
        return config


def _tf_setup():
    tf.set_random_seed(random.randint(0, 1000))


def _tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4 
    return config


def _perplexity(loss):
    return np.exp(loss)


class LanguageModel(object):

    def __init__(self, batch_size, num_steps, vocab_size):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.vocab_size = vocab_size

        self.saver = None
        self.session = None

        # Input placeholders
        self.word_ids = None
        self.label_ids = None
        self.word_embeddings = None
        self.is_training = None

        # Output tensors
        self.loss = None
        self.top_layer = None

        # Optimisation op
        self.learning_rate = None
        self.optimise = None

    def construct_network(self, hidden_dims, num_layers, dropout_keep_prob,
                          max_gradient_norm):
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")

        # (batch size x num steps)
        self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
        self.label_ids = tf.placeholder(tf.int32, [None, None], name="label_ids")

        # (vocab size x hidden dims)
        self.word_embeddings = tf.get_variable(
                "word_embeddings",
                shape=[self.vocab_size, hidden_dims])

        # (batch size x num steps x embedding dims)
        input_tensor = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids)

        # Conditionally set dropout keep probability depending on whether we
        # are training at the moment. A prob of 1.0 will mean no dropout is
        # applied.
        dropout_keep_prob = tf.cond(self.is_training,
                lambda: tf.constant(dropout_keep_prob),
                lambda: tf.constant(1.0))

        input_tensor = tf.nn.dropout(input_tensor, dropout_keep_prob)

        inputs = tf.unstack(input_tensor, num=self.num_steps, axis=1)

        cell = tf.contrib.rnn.LSTMBlockCell(
                hidden_dims,
                forget_bias=0.0)

        cell =  tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell(
                [cell for _ in range(num_layers)],
                state_is_tuple=True)

        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        outputs, self.final_state = tf.nn.static_rnn(
                cell,
                inputs,
                dtype=tf.float32,
                initial_state=self.initial_state)

        # ((batch size * num steps) x hidden dims) 
        output_tensor = tf.reshape(tf.concat(outputs, 1), [-1, hidden_dims])
        self.top_layer = output_tensor

        # ((batch size * num steps) x vocab) 
        logits = tf.layers.dense(output_tensor, self.vocab_size,
                                 name='project_onto_vocab')

        # (batch size x num steps x vocab)
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, self.vocab_size])

        loss = tf.contrib.seq2seq.sequence_loss(
                logits=logits,
                targets=self.label_ids,
                weights=tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
                average_across_timesteps=False,
                average_across_batch=True)

        self.loss = tf.reduce_sum(loss)

        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          max_gradient_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.optimise = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step())

        self.saver = tf.train.Saver()

    def save(self, save_path):
        logger.info('Saving model to %s', save_path)
        self.saver.save(self.session, save_path)

    def load(self, load_path):
        logger.info('Loading model from %s', load_path)
        self.saver.restore(self.session, load_path)

    def preload_word_embeddings(self, embedding_path, vocabulary, expected_dims):
        logger.info('Preloading embeddings from %s', embedding_path)

        embedding_matrix = self.session.run(self.word_embeddings)

        with open(embedding_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip the first line - it's metadata
            for line in f:
                w, *v = line.strip().split()

                w = data.preprocess_token(w)

                if len(v) != expected_dims:
                    logger.warning('Embedding for "%s" has %d dimensions, but '
                                   'expected %d. Skipping.',
                                   w, len(v), expected_dims)
                    continue

                if w in vocabulary:
                    embedding_matrix[vocabulary.to_id(w)] = v

        self.session.run(self.word_embeddings.assign(embedding_matrix))

        logger.info('Finished loading embeddings')

    def run_batch(self, feed_dict, is_training=False):
        if is_training:
            cost, final_state, _ = self.session.run(
                [self.loss, self.final_state, self.optimise],
                feed_dict)
            top_layer = None
        else:
            cost, final_state, top_layer  = self.session.run(
                [self.loss, self.final_state, self.top_layer],
                feed_dict)

        return cost, final_state, top_layer

    def run_epoch(self, x_y, learning_rate=1.0, is_training=False):
        state = self.session.run(self.initial_state)

        cost = 0.0
        iters = 0
        top_layers = []
        for x_batch, y_batch in x_y:
            feed = {
                self.word_ids: x_batch,
                self.label_ids: y_batch,
                self.learning_rate: learning_rate,
                self.is_training: is_training,
            }

            for i, (c, h) in enumerate(self.initial_state):
                feed[c] = state[i].c
                feed[h] = state[i].h

            c, state, top_layer = self.run_batch(feed, is_training)
            top_layers.append(top_layer)

            cost += c
            iters += self.num_steps
        
        return cost / iters, top_layers


def _train(lm, train_x_y, dev_x_y=None, save_path=None,
           num_epochs=10, terminate_after=5, learning_rate_decay=1.0, decay_after=None):
    logger.info('Training...')
    best_epoch = 0
    best_epoch_score = math.inf
    num_epochs_without_improvement = 0
    for epoch in range(num_epochs):
        train_x_y, train_x_y_actual = itertools.tee(train_x_y)
        dev_x_y, dev_x_y_actual = itertools.tee(dev_x_y) if dev_x_y else (None, None)

        logger.info('Epoch: %d', epoch)

        # Assume an initial learning rate of 1.0, implicit in this calculation
        learning_rate = learning_rate_decay ** max(epoch + 1 - decay_after, 0.0)
        logger.info('Learning rate: %f', learning_rate)

        loss, _ = lm.run_epoch(train_x_y_actual, learning_rate=learning_rate, is_training=True)
        perplexity = _perplexity(loss)
        logger.info('Training perplexity: %f', perplexity)

        if dev_x_y:
            loss, _ = lm.run_epoch(dev_x_y_actual, is_training=False)
            logger.info('Validation perplexity: %f', _perplexity(loss))

        if perplexity - best_epoch_score < -1:
            logger.info('Best epoch updated to: %d', epoch)
            best_epoch = epoch
            best_epoch_score = perplexity
            num_epochs_without_improvement = 0
            if save_path:
                lm.save(save_path)
        else:
            logger.info('Best epoch: %d', best_epoch)
            num_epochs_without_improvement += 1 

        if num_epochs_without_improvement >= terminate_after:
            logger.info('Terminating')
            logger.info('Best epoch is %d with %d', best_epoch, best_epoch_score)
            break


def _test(lm, x_y):
    logger.info('Testing...')

    loss, _ = lm.run_epoch(x_y, is_training=False)
    logger.info('Perplexity: %f', _perplexity(loss))


def _annotate(lm, x_y):
    logger.info('Annotating...')

    loss, top_layers = lm.run_epoch(x_y, is_training=False)
    logger.info('Perplexity: %f', _perplexity(loss))

    return np.concatenate(top_layers, axis=0)
    

def _main():
    args = _parse_args()
    config = _load_config(args.config_path)
    _tf_setup()

    # Parse args and allocate training and test data
    if args.train and args.test is None:
        logger.info('Splitting training data into training and test sets')
        training_data, test_data = model_selection.train_test_split(
            list(data.read_path(args.train[0])),
            test_size=0.1)
    else:
        training_data = data.read_path(args.train[0]) if args.train else None
        test_data = data.read_path(args.test) if args.test else None

    if args.validate:
        dev_data = data.read_path(args.validate)
    elif args.validate is None:
        logger.info('Splitting training data into training and dev sets')
        training_data, dev_data = model_selection.train_test_split(
            list(training_data),
            test_size=0.1)
    else:
        dev_data = None

    training_data, vocab_data = itertools.tee(training_data)

    # Build (or load) the vocabulary
    v = vocab.Vocabulary()
    if args.load_vocab:
        v.load(args.load_vocab)
    else:
        v.build(vocab_data)
        if args.save_vocab:
            v.save(args.save_vocab)

    # Set up the language model
    lm = LanguageModel(
            batch_size=config['data']['batch_size'],
            num_steps=config['data']['num_steps'],
            vocab_size=v.size)

    lm.construct_network(
            hidden_dims=config['lstm']['hidden_dims'],
            num_layers=config['lstm']['num_layers'],
            dropout_keep_prob=config['lstm']['dropout']['keep_prob'],
            max_gradient_norm=config['lstm']['max_gradient_norm'])

    training_data = v.to_ids(training_data)
    dev_data = v.to_ids(dev_data)
    test_data = v.to_ids(test_data)

    with tf.Session(config=_tf_config()) as sess:
        lm.session = sess
        # We just do this to set a global default initializer
        with tf.variable_scope('Top',
                               initializer=tf.random_uniform_initializer(
                                   -config['lstm']['init_scale'],
                                   config['lstm']['init_scale'])):

            lm.session.run(tf.global_variables_initializer())

            # Optionally restore a model from file - note that this relies on loading
            # the right vocab, too
            if args.load: 
                lm.load(args.load)
            elif args.embeddings:
                lm.preload_word_embeddings(args.embeddings, v, config['lstm']['hidden_dims'])

            if args.train:
                train_batches = data.batch_data(training_data,
                        batch_size=config['data']['batch_size'],
                        num_steps=config['data']['num_steps'])

                if dev_data:
                    dev_batches = data.batch_data(dev_data,
                            batch_size=config['data']['batch_size'],
                            num_steps=config['data']['num_steps'])
                else:
                    dev_batches = None

                _train(lm, train_batches, dev_batches, args.save,
                       num_epochs=config['training']['num_epochs'],
                       terminate_after=config['training']['terminate_after'],
                       learning_rate_decay=config['training']['learning_rate_decay'],
                       decay_after=config['training']['decay_after'])

            if args.test is not False:
                test_batches = data.batch_data(test_data,
                        batch_size=config['data']['batch_size'],
                        num_steps=config['data']['num_steps'])
                _test(lm, test_batches)

            if args.annotate is not False:
                input_path, output_path = args.annotate
                token_label_pairs = annotate.read_file(input_path)
                to_annotate = v.to_ids(annotate.pairs_to_lm_format(token_label_pairs))
                to_annotate_batches = data.batch_data(to_annotate,
                        batch_size=config['data']['batch_size'],
                        num_steps=config['data']['num_steps'])
                top_layers = _annotate(lm, to_annotate_batches)

                annotate.write_file(output_path, token_label_pairs, top_layers,
                                    config['lstm']['hidden_dims'])


if __name__ == '__main__':
    _main()
