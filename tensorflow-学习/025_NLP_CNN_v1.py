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

(x_train, y_train),(x_test, y_test) = keras.datasets.imdb.load_data()

print(x_train.shape)
print(x_train[1])