# -*- coding: utf-8 -*

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape((-1,28,28,1))
x_test = x_test.reshape((-1, 28,28,1))

x_shape = x_train.shape

deep_model = keras.Sequential(
    [
        #layers.Conv2D(input_shape = ((x_shape[1], x_shape[2], x_shape[3])), filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
        layers.Conv2D(input_shape = ((x_shape[1], x_shape[2], x_shape[3])), filters = 32, kernel_size = (3,3), strides = (1,1), padding = 'same',activation = 'relu'),
        layers.BatchNormalization(),
        layers.Conv2D(filters = 16, kernel_size = (1,1),strides = (1,1),padding = 'valid', activation = 'relu'),
        layers.Conv2D(filters = 32, kernel_size= (3,3),strides = (1,1), padding = 'same',activation = 'relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size = (2,2)),
        layers.Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), padding = 'same', activation = 'relu'),
        layers.Conv2D(filters = 16, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu'),
        layers.MaxPool2D(pool_size = (2,2)),
        layers.Flatten(),
        layers.Dense(32,activation = 'relu'),
        layers.Dense(10, activation = 'softmax')
    ]
)

deep_model.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.SparseCategoricalCrossentropy(),
metrics = ['accuracy'])
print(deep_model.summary())


history = deep_model.fit(x_train, y_train, batch_size = 64, epochs = 5, validation_split = 0.1)

result = deep_model.evaluate(x_test, y_test)

print(result)