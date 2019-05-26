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


word_index = imdb.get_word_index(
    path='/Users/zhangpeng/Downloads/DataSets/imdb/imdb_word_index.json')

print('s')
word2id = {}
for k, v in word_index.items():
    word2id[k] = v+3

word2id['<PAD>'] = 0
word2id['<START>'] = 1
word2id['<UNK>'] = 2
word2id['<UNUSED>'] = 3

#print(word2id.keys())

id2word = {}

for k,v in word_index.items():
    id2word[v] = k

def get_words(sent_ids):
    all_word = ''
    for i in sent_ids:
        one_word = id2word.get(i, '?')
        print(one_word)
        all_word = all_word + ' ' + one_word
    return all_word

sent = get_words(train_x[0])
print(sent)



id2word = {v: k for k, v in word2id.items()}
def get_words(sent_ids):
    print([id2word.get(i, '?') for i in sent_ids])
    return ' '.join([id2word.get(i, '?') for i in sent_ids])


sent = get_words(train_x[0])
print(sent)
