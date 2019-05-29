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

x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])

model = keras.Sequential(
    [
        layers.Dense(64, activation = 'relu', input_shape = (784,)),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(64, activation = 'relu'),
        layers.Dense(10, activation = 'softmax')
    ]
)
model.compile(optimizer = keras.optimizers.Adam(),
              loss = keras.losses.SparseCategoricalCrossentropy(),
              metrics = ['accuracy'])

print(model.summary())

#history = model.fit(x_train, y_train, batch_size=256,epochs=100, validation_split=0.3, verbose=1)S
#esult = model.evaluate(x_test, y_test)
#print(model.metrics_names)
#print(result)


model = keras.Sequential(
    [
        layers.Dense(64, activation='relu',
                     kernel_initializer='he_normal', input_shape=(784,)),
        layers.Dense(64, activation='relu',
                     kernel_initializer='he_normal'),
        layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
        layers.Dense(10, activation = 'softmax')
    ]
)

model.compile(optimizer = keras.optimizers.Adam(),
              loss = keras.losses.SparseCategoricalCrossentropy(),
              metrics = ['accuracy'])

#history = model.fit(x_train, y_train, batch_size = 256, epochs = 100, validation_split = 0.3, verbose = 1)
#print(history)
#result = model.evaluate(x_test, y_test)
#print(result)


model = keras.Sequential([
    layers.Dense(64, activation = 'sigmoid', input_shape = (784,)),
    layers.BatchNormalization(),
    layers.Dense(64, activation = 'sigmoid'),
    layers.BatchNormalization(),
    layers.Dense(64, activation = 'sigmoid'),
    layers.BatchNormalization(),
    layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer =keras.optimizers.Adam(),
              loss = keras.losses.SparseCategoricalCrossentropy(),
              metrics = ['accuracy'])

print(model.summary())

history = model.fit(x_train, y_train, batch_size = 256, epochs = 100, validation_split = 0.3, verbose = 1)
print(history.history)

reslut = model.evaluate(x_test, y_test)

print(reslut)

