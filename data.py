import itertools
import logging
import re

import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_SENTENCE_BOUNDARY = '<s>'


def read_file(path):
    """Reads input files in whitespace-separated format.

    Strips punctutation when from line ends. Replaces line breaks with sentence
    boundary marker. Preprocesses each token.

    Returns a single list of tokens.
    """
    logger.info('Reading data from %s', path)
    with tf.gfile.GFile(path, "r") as f:
        return preprocess_text(f.read())


def batch_data(data, batch_size=20, num_steps=35):
    """Batch data into processable chunks.

    Returns two matrices of dimensions
    (batches per epoch x batch size x num steps). The first is the data, and
    the second is the data shifted on by one, to form the labels.
    """
    logger.info('Batching data comprising %d elements', len(data))

    batches_per_epoch = ((len(data) // batch_size) - 1) // num_steps
    target_length = batches_per_epoch * batch_size * num_steps
    num_repetitions = (target_length // len(data)) + 1

    repeated = list(itertools.chain.from_iterable(
        itertools.repeat(data, num_repetitions)))[:target_length]

    x = np.reshape(repeated, (batches_per_epoch, batch_size, num_steps))

    shift = data[1:] + [data[0]]
    repeated_shift = list(itertools.chain.from_iterable(
        itertools.repeat(shift, num_repetitions)))[:target_length]

    y = np.reshape(repeated_shift, (batches_per_epoch, batch_size, num_steps))

    logger.info('Created two (%d x %d x %d) matrices',
                batches_per_epoch, batch_size, num_steps)

    return x, y


def preprocess_text(text):
    no_ending_punct = re.sub(r' [\.\,]\n', '\n', text)
    with_sentence_bounds = no_ending_punct.replace(
        '\n', ' {} '.format(_SENTENCE_BOUNDARY))
    return preprocess_tokens(with_sentence_bounds.split())


def preprocess_tokens(tokens):
    return [preprocess_token(t) for t in tokens]


def preprocess_token(token):
    return re.sub(r'\d', '0', token.lower())
