# tf-lstm-lm

A Tensorflow implementation of a undirectional, multilayer LSTM language model,
loosely following Zaremba et al., 2014 [1].

After training a model, it can be used to annotate text with
timestep-by-timestep LSTM output vectors, for usage in downstream tasks (e.g.
error detection or sequence labelling).

I've taken heavy inspiration from [2, 3, 4]. In development, I used Mikolov's
tokenised PTB dataset [5].

## Dependencies
* Python 3.6
* pipenv

## Setup
```
pipenv install

```

## Usage

### Training and evaluation
```
pipenv run python lstm.py conf/test.yaml
    --train data/ptb.train.txt
    --validate data/ptb.valid.txt
    --test data/ptb.test.txt
```

If you don't wish to specify a test or devtest set, the training data will be
split to serve both:
```
pipenv run python lstm.py conf/test.yaml
    --train data/ptb.train.txt
    --validate
    --test
```

### Streaming
For large datasets, it might be useful to stream shards of data from disk. To
do so, pass a wildcard path:

```
pipenv run python lstm.py conf/test.yaml
    --train data/shards/train/*.txt
    --validate data/shards/dev/*.txt
    --test data/shards/test/*.txt
```
Note that it is not possible to split streams into dev and test sets: you'll
need to specify files or paths if you wish to validate and test.

### Saving and loading
To save a model:
```
pipenv run python lstm.py conf/test.yaml
    --train data/ptb.train.txt
    --save models/foo.ckpt
    --save_vocab models/foo.vocab
```

To load a pre-saved model:
```
pipenv run python lstm.py conf/test.yaml
    --test data/ptb.train.txt
    --load models/foo.ckpt
    --load_vocab models/foo.vocab
```

### Annotation
Given a pre-trained model, and TSV file of token-label pairs, with a single
pair on each line, it is possible to annotate each line with an LSTM output
vector corresponding to the token.

For example, take the following file for an error detection task:
```
The	c
cat	c
sat	c
on	c
the	c
the	i
mat	c
.	c
```

where we'd like to annotate the file with hidden state as follows (note that
this example uses a trivially small number of dimensions):
```
The	c	1.0	2.0	3.0
cat	c	2.0	2.0	3.0
sat	c	1.0	1.0	3.0
on	c	3.0	2.0	3.0
the	c	3.0	2.0	3.0
the	i	1.0	1.0	3.0
mat	c	3.0	2.0	1.0
.	c	1.0	2.0	3.0
```

we can do this as follows:
```
pipenv run python lstm.py conf/test.yaml
    --load models/foo.ckpt
    --load_vocab models/foo.vocab
    --annotate data/input.tsv data/output.tsv
```

## References
[1] [Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent neural network regularization.](https://arxiv.org/abs/1409.2329)
[2] [https://github.com/wpm/tfrnnlm/](https://github.com/wpm/tfrnnlm/)
[3] [https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/](https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/)
[4] [https://www.tensorflow.org/tutorials/recurrent](https://www.tensorflow.org/tutorials/recurrent)
[5] [http://www.fit.vutbr.cz/~imikolov/rnnlm/](http://www.fit.vutbr.cz/~imikolov/rnnlm/)

