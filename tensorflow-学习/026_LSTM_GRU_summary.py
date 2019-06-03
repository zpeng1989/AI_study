# -*- coding: utf-8 -*

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

num_words = 30000
maxlen = 200

#(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words = num_words)

#x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen, padding='post')
#x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen, padding='post')

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words = num_words)

print(x_train.shape)


x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen, padding = 'post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen, padding = 'post')


def lstm_model():
    model = keras.Sequential([
        layers.Embedding(input_dim = 30000, output_dim = 32, input_length = maxlen),
        layers.LSTM(32, return_sequences = True),
        layers.LSTM(1, activation = 'sigmoid', return_sequences = False)
    ])
    model.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
    return model

model = lstm_model()

print(model.summary())

history = model.fit(x_train, y_train, batch_size = 64, epochs = 5, validation_split = 0.1)


def lstm_model():
    model = keras.Sequential([
        layers.Embedding(input_dim = 30000, output_dim = 32, input_length = maxlen),
        layers.GRU(32, return_sequences = True),
        layers.GRU(1, activation = 'sigmoid', return_sequences = False)
    ])
    model.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
    return model

model = lstm_model()
model.summary()

history = model.fit(x_train, y_train, batch_size = 64, epochs = 5, validation_split = 0.1)






