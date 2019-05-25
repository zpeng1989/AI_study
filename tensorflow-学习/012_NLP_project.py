# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.keras.backend.clear_session()
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


imdb = keras.datasets.imdb

(train_x, train_y),(test_x, test_y) = keras.datasets.imdb.load_data()

print(len(train_x))
print(len(test_x))
print(len(train_y))


