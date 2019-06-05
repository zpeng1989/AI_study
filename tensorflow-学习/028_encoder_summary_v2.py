# -*- coding: utf-8 -*

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = tf.expand_dims(x_train.astype('float32'), -1) / 255.0
x_test = tf.expand_dims(x_test.astype('float32'), -1) / 255.0

print(x_train.shape)

inputs = layers.Input(shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]), name = 'inputs')
code = layers.Conv2D(16,(3,3), activation = 'relu', padding = 'same')(inputs)
code = layers.MaxPool2D((2,2), padding = 'same')(code)
print(code.shape)

decode = layers.Conv2D(16, (3,3), activation = 'relu', padding = 'same')(code)
decode = layers.UpSampling2D((2,2))(decode)
print(decode.shape)
outputs = layers.Conv2D(1, (3,3), activation = 'sigmoid', padding = 'same')(decode)
print(outputs.shape)
auto_encoder = keras.Model(inputs, outputs)

auto_encoder.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy())

print(auto_encoder.summary())

auto_encoder.fit(x_train, x_train, batch_size = 64, epochs = 5, validation_freq = 10)



