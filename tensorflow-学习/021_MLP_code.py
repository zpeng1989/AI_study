# -*- coding: utf-8 -*

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as p
'''
(x_train, y_train),(x_test, y_test) = keras.datasets.boston_housing.load_data()
print(x_train.shape, " ", y_train.shape)
print(x_test.shape, " ", y_test.shape)

model = keras.Sequential(
    [
        layers.Dense(32, activation = 'sigmoid',input_shape = (13,)),
        layers.Dense(32, activation = 'sigmoid'),
        layers.Dense(32, activation = 'sigmoid'),
        layers.Dense(1)
    ]
)

model.compile(optimizer = keras.optimizers.SGD(0.1),
              loss = 'mean_squared_error',
              metrics = ['mse']
                )

print(model.summary())

model.fit(x_train, y_train, batch_size = 50, epochs = 50, validation_split = 0.1, verbose = 1)

result = model.evaluate(x_test, y_test)

print(model.metrics_names)
print(result)
'''
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()
x_data = whole_data.data
y_data = whole_data.target

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3)

print(x_train.shape, " ", y_train.shape)
print(x_test.shape, " ", y_test.shape)

model = keras.Sequential(
    [
        layers.Dense(32, activation='sigmoid', input_shape=(30,)),
        layers.Dense(32, activation='sigmoid'),
        layers.Dense(32, activation='sigmoid'),
        layers.Dense(1)
    ]
)

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss='mean_squared_error',
              metrics=['mse']
              )

print(model.summary())

model.fit(x_train, y_train, batch_size=50,
          epochs=50, validation_split=0.1, verbose=1)

result = model.evaluate(x_test, y_test)

print(model.metrics_names)
print(result)
