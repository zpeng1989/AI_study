# -*- coding: utf-8 -*

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()

print(x_train.shape)
print(x_test.shape)

model.compile(optimaizer = keras.optimizers.SGD(0.1),
              loss = 'mean_squared_error',
              metrics = ['mse'])

print(model.summary())

model.fit(x_train, y_train, batch_size = 50, epochs = 50, validation_split = 0.1, verbose = 1)

result = model.evaluate(x_test, y_test)

print(model.metrics_names)
print(result)



