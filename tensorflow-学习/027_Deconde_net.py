# -*- coding: utf-8 -*

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape((-1, 28*28))/255.0
x_test = x_test.reshape((-1, 28*28))/255.0

#print(x_train[1])

print(x_train.shape)

code_dim = 32

inputs = layers.Input(shape = (x_train.shape[1],), name= 'inputs')
code = layers.Dense(code_dim, activation = 'relu', name = 'code')(inputs)
outputs = layers.Dense(x_train.shape[1], activation = 'softmax', name = 'outps')(code)

auto_encoder = keras.Model(inputs, outputs)
auto_encoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
print(auto_encoder.summary())

history = auto_encoder.fit(x_train, x_train, batch_size = 64, epochs = 5,validation_split = 0.1)
