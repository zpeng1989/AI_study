# -*- coding: utf-8 -*

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences




vcoab_size = 10000
(train_x, train_y), (test_x, test_y) = keras.datasets.imdb.load_data(num_words = vcoab_size)
print(train_x[0])

word_index = keras.datasets.imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3
reverse_word_index = {v:k for k, v in word_index.items()}
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(train_x[0]))

maxlen = 500
train_x = keras.preprocessing.sequence.pad_sequences(train_x, value = word_index['<PAD>'], padding = 'post', maxlen = maxlen)
test_x = keras.preprocessing.sequence.pad_sequences(test_x, value = word_index['<PAD>'], padding = 'post', maxlen = maxlen)

embedding_dim = 16
model = keras.Sequential([
    layers.Embedding(vcoab_size, embedding_dim, input_length = maxlen),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')
])


print(model.summary())


model.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
history = model.fit(train_x, train_y, epochs = 5, batch_size = 512, validation_split = 0.1)


e = model.layers[0]

weights = e.get_weights()[0]
print(weights.shape)

out_v = open('vecs.tsv', 'w')
out_m = open('meta.tsv', 'w')
for word_num in range(vcoab_size):
    word = reverse_word_index[word_num]
    embeddings = weights[word_num]
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()
