from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow.keras.layers as layers

import time
import numpy as np

print(tf.__version__)


examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info = True, as_supervised = True)

train_examples, val_examples = examples['train'], examples['validation']

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((en.numpy() for pt, en in train_examples), target_vorcab_size= 2**13)
tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus((pt.numpy() for pt, en in train_examples), target_vorcab_size=2**13)

sample_str = 'hello world, tensorflow 2'

tokenized_str = tokenizer_en.encode(sample_str)
print(tokenized_str)
original_str = tokenizer_en.decode(tokenized_str)
print(original_str)


def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(lang1.numpy()) + [tokenizer_pt.vocab_size + 1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(lang1.numpy()) + [tokenizer_en.vocab_size + 1]
    return lang1, lang2

MAX_LENGTH = 40
def filter_long_sent(x,y, max_length = MAX_LENGTH):
    return rf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

def tf_encode(pt, en):
    return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

BUFFER_SIZE = 20000
BATCH_SIZE = 64

train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_long_sent)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.padded_batch(BATCH_SIZE, padded_shapes = ([40], [40]))
train_dataset = train_dataset.prefet(tf.data.experimental.AUTOTUNE)

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_long_sent).padded_batch(BATCH_SIZE, padded_shapes = ([40], [40]))
de_batch, en_batch = next(iter(train_dataset))


def get_angles(pos, i, d_model):
    angle_rates = 1/np.power(10000, (2*(i//2))/np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis,:], d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cones = np.cos(angle_rads[:, 1::2])
    

