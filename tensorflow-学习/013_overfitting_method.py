# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.keras.backend.clear_session()
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np


NUM_WORDS = 10000
(train_data,train_labels),(test_data,test_labels)= keras.datasets.imdb.load_data(num_words = NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences),dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0
    return results

train_data = multi_hot_sequences(train_data, dimension = NUM_WORDS)
test_data = multi_hot_sequences(test_data,dimension = NUM_WORDS)

import tensorflow.keras.layers as layers

baseline_model = keras.Sequential(
    [
        layers.Dense(16, activation = 'relu', input_shape = (NUM_WORDS,)),
        layers.Dense(16, activation = 'relu'),
        layers.Dense(1, activation = 'sigmoid')
    ]
)

baseline_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', 'binary_crossentropy'])
print(baseline_model.summary())

baseline_history = baseline_model.fit(train_data,train_labels,epochs = 20, batch_size = 512, validation_data = (test_data,test_labels),verbose = 1)