# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt

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
#baseline_history = baseline_model.fit(train_data,train_labels,epochs = 5, batch_size = 512, validation_data = (test_data,test_labels),verbose = 1)

small_model = keras.Sequential(
    [
        layers.Dense(4, activation = 'relu', input_shape = (NUM_WORDS,)),
        layers.Dense(4, activation = 'relu'),
        layers.Dense(1, activation = 'sigmoid')
    ]
)

small_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', 'binary_crossentropy'])
print(small_model.summary())
#samll_history = small_model.fit(train_data,train_labels, epochs = 5, batch_size = 512, validation_data = (test_data, test_labels),verbose = 1)


big_model = keras.Sequential(
    [
        layers.Dense(512, activation='relu', input_shape=(NUM_WORDS,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ]
)

big_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
                    'accuracy', 'binary_crossentropy'])
print(big_model.summary())
#big_history = big_model.fit(train_data, train_labels, epochs=5,
#                                batch_size=512, validation_data=(test_data, test_labels), verbose=1)

def plot_history(histories, key = 'binary_crossentropy'):
    plt.figure(figsize = (16,10))
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key], '--', label = name.title()+ 'Val')
        plt.plot(history.epoch, history.history[key], color = val[0].get_color(),label = name.title() + 'Train')
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])

#plot_history([('baseline', baseline_history),('small', samll_history),('big', big_history)])

#plt.show()

l2_model = keras.Sequential(
    [
        layers.Dense(16, kernel_regularizer = keras.regularizers.l2(0.01), activation = 'relu', input_shape = (NUM_WORDS,)),
        layers.Dense(16, kernel_regularizer = keras.regularizers.l2(0.01), activation = 'relu'),
        layers.Dense(1, activation = 'sigmoid')
    ]
)

l2_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', 'binary_crossentropy'])

print(l2_model.summary())

l2_history = l2_model.fit(train_data, train_labels,epochs = 10, batch_size = 512, validation_data = (test_data, test_labels))
