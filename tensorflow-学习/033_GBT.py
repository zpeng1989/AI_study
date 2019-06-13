from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from IPython.display import clear_output

# Load dataset.
dftrain = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

print(dftrain.head())
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

import tensorflow as tf

tf.random.set_seed(123)

print(dftrain.describe())




