import glob
import itertools
import logging
import re

import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_SENTENCE_BOUNDARY = '<s>'


def read_path(path):
    """Reads input files in whitespace-separated format.

    Will expand globbed paths.

    Strips punctutation when from line ends. Replaces line breaks with sentence
    boundary marker. Preprocesses each token.

    Returns a single list of tokens.
    """
    def read(p):
        logger.info('Reading data from %s', p)
        with open(p, 'r') as f:
            return preprocess_text(f.read())

    return itertools.chain.from_iterable(read(p) for p in glob.glob(path))


def batch_data(data, batch_size=20, num_steps=35):
    """Batch data into processable chunks.

    Returns a generator of (x, y) tuples, where each x is a
    (batch_size x num_steps) matrix of data, and y is a matrix of labels with
    the same dimensions.
    """
    start_index = 0
    while True:
        end_index = start_index + (batch_size * num_steps)

        slice_of_tokens = list(itertools.islice(data, start_index, end_index))

        if len(slice_of_tokens) != (batch_size * num_steps):
            # TODO: This does mean we might break before processing the
            # remnants of the last batch, if it doesn't evenly divide. Perhaps
            # we should use some padding?
            break

        shift = slice_of_tokens[1:] + [slice_of_tokens[0]]
        x = np.reshape(slice_of_tokens, (batch_size, num_steps))
        y = np.reshape(shift, (batch_size, num_steps))

        yield x, y

        start_index += end_index


def preprocess_text(text):
    no_ending_punct = re.sub(r' [\.\,]\n', '\n', text)
    with_sentence_bounds = no_ending_punct.replace(
        '\n', ' {} '.format(_SENTENCE_BOUNDARY))
    return preprocess_tokens(with_sentence_bounds.split())


def preprocess_tokens(tokens):
    return [preprocess_token(t) for t in tokens]


def preprocess_token(token):
    return re.sub(r'\d', '0', token.lower())
