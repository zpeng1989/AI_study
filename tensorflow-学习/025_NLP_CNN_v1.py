# -*- coding: utf-8 -*

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences


num_features = 3000
sequence_length = 300
embedding_dimension = 100

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
    num_words=num_features)

print(x_train.shape)
print(x_train[1])
print(len(x_train[1]))

x_train = pad_sequences(x_train, maxlen = sequence_length)
x_test = pad_sequences(x_test, maxlen = sequence_length)

print(x_train.shape)
print(x_train[1])


def imdb_cnn():
    model = keras.Sequential([
        layers.Embedding(input_dim = num_features, output_dim = embedding_dimension, input_length = sequence_length),
        layers.Conv1D(filters = 50, kernel_size = 5, strides = 1, padding = 'valid'),
        layers.MaxPool1D(2, padding = 'valid'),
        layers.Flatten(),
        layers.Dense(10, activation = 'relu'),
        layers.Dense(1, activation = 'sigmoid')
    ])
    model.compile(optimizer = keras.optimizers.Adam(1e-3), loss = keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
    return model

model = imdb_cnn()

print(model.summary())

#history = model.fit(x_train, y_train, batch_size = 64, epochs = 5, validation_split = 0.1)



filter_size = [3,4,5]

def convolution():
    inn = layers.Input(shape = (sequence_length, embedding_dimension, 1))
    cnns = []
    for size in filter_size:
        conv = layers.Conv2D(filters = 64, kernel_size = (size, embedding_dimension), strides = 1, padding = 'valid', activation = 'relu')(inn)
        pool = layers.MaxPool2D(pool_size = (sequence_length-size+1,1),padding = 'valid')(conv)
        cnns.append(pool)
    outt = layers.concatenate(cnns)
    model = keras.Model(inputs = inn, outputs =outt)
    #model = keras.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
    return model

def cnn_mulfilter():
    model = keras.Sequential([
        layers.Embedding(input_dim = num_features, output_dim = embedding_dimension, input_length = sequence_length),
        layers.Reshape((sequence_length, embedding_dimension, 1)),
        convolution(),
        layers.Flatten(),
        layers.Dense(10, activation = 'relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation = 'sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(
    ), loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model

model = cnn_mulfilter()
print(model.summary())

history = model.fit(x_train, y_train, batch_size = 64, epochs = 5, validation_split = 0.1)






