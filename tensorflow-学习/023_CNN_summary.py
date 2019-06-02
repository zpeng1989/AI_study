# -*- coding: utf-8 -*

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(x_train.shape)
print(x_test.shape)


x_train = x_train.reshape((-1,28,28,1))
x_test = x_test.reshape((-1,28,28,1))

model = keras.Sequential()

model.add(layers.Conv2D(input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3]),filters = 32, kernel_size = (3,3), strides = (1,1), padding = 'valid', activation = 'relu'))

model.add(layers.MaxPool2D(pool_size = (2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(32, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])


print(model.summary())

history = model.fit(x_train, y_train, batch_size = 64, epochs = 5, validation_split = 0.1)

res = model.evaluate(x_test, y_test)

print(res)
