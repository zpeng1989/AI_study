# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.keras.backend.clear_session()
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


imdb = keras.datasets.imdb

(train_x, train_y), (test_x, test_y) = keras.datasets.imdb.load_data(num_words=10000)

print(len(train_x))
print(len(test_x))
print(len(train_y))


#word_index = imdb.get_word_index(
#    path='/Users/zhangpeng/Downloads/DataSets/imdb/imdb_word_index.json')

word_index = imdb.get_word_index()

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

for k,v in word2id.items():
    id2word[v] = k

#id2word = {v: k for k, v in word2id.items()}
def get_words(sent_ids):
    all_word = ''
    for i in sent_ids:
        one_word = id2word.get(i, '?')
        #print(i)
        #print(id2word.get(i,'?'))
        #print(id2word[i])
        #print(one_word)
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

train_x = keras.preprocessing.sequence.pad_sequences(
    train_x, value = word2id['<PAD>'], padding = 'post', maxlen = 256
)

test_x = keras.preprocessing.sequence.pad_sequences(
    test_x, value = word2id['<PAD>'], padding = 'post', maxlen = 256
)

print(train_x[0])
print(len(train_x[0]))
print(len(train_x[1]))


vocab_size = 20000

model = keras.Sequential()
model.add(layers.Embedding(vocab_size, 16))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
print(model.summary())

'''
imdb = keras.datasets.imdb
(train_x, train_y), (test_x, text_y) = keras.datasets.imdb.load_data(num_words=10000)
'''
word_index = imdb.get_word_index()

word2id = {k: (v+3) for k, v in word_index.items()}
word2id['<PAD>'] = 0
word2id['<START>'] = 1
word2id['<UNK>'] = 2
word2id['<UNUSED>'] = 3

id2word = {v: k for k, v in word2id.items()}


def get_words(sent_ids):
    return ' '.join([id2word.get(i, '?') for i in sent_ids])


sent = get_words(train_x[0])
print(sent)


# 句子末尾padding
train_x = keras.preprocessing.sequence.pad_sequences(
    train_x, value=word2id['<PAD>'],
    padding='post', maxlen=256
)
test_x = keras.preprocessing.sequence.pad_sequences(
    test_x, value=word2id['<PAD>'],
    padding='post', maxlen=256
)
print(train_x[0])
print('len: ', len(train_x[0]), len(train_x[1]))


vocab_size = 10000
model = keras.Sequential()
model.add(layers.Embedding(vocab_size, 16))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])










x_val = train_x[:10000]
x_train = train_x[10000:]

y_val = train_y[:10000]
y_train = train_y[10000:]

history = model.fit(x_train, y_train,
                    epochs=40, batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

result = model.evaluate(test_x, text_y)
print(result)


x_val = train_x[:10000]
x_train = train_x[10000:]

y_val = train_y[:10000]
y_train = train_y[10000:]

print(len(train_x))
print(y_val[:3])

history = model.fit(x_train, y_train, epochs = 40, batch_size = 512, validation_data = (x_val, y_val),verbose = 1)
result = model.evaluate(test_x, test_y)
print(result)
