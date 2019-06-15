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
fc = tf.feature_column

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

def one_hot_cat_column(feature_name, vocab):
    return tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocab))

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vovabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vovabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype = tf.float32))

print(feature_columns)

example = dict(dftrain.head(1))
class_fc = tf.feature_column.indicator_column(
    tf.feature_column.categorical_column_with_vocabulary_list('class', ('First', 'Second', 'Third')))


print('Feature value: "{}"'.format(example['class'].iloc[0]))
print('One-hot encoded: ', tf.keras.layers.DenseFeatures([class_fc])(example).numpy())


NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs = None, shuffle = True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        dataset = dataset.repeat(n_epochs)
        dataset = dataset.batch(NUM_EXAMPLES)
        return dataset
    return input_fn


train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle = Flase, n_epochs = 1)



linear_est = tf.estimator.LinearClassifier(feature_columns)

linear_est.train(train_input_fn, max_steps = 10000)

result = linear_est.evaluate(eval_input_fn)
clear_output()

print(result)

n_batches = 1

est = tf.estimator.BoostedTreesClassifier(feature_columns, n_batches_per_layer = n_batches)

est.train(train_input_fn, max_steps = 100)

pred_dicts = list(est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])


