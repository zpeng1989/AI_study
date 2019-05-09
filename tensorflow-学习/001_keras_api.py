import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
#import keras

inputs = tf.keras.Input(shape = (784,), name = 'img')
h1 = layers.Dense(32, activation = 'relu')(inputs)
h2 = layers.Dense(32, activation = 'relu')(h1)
outputs = layers.Dense(10, activation = 'softmax')(h2)
model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'mnist model')

print(model.summary())

#tf.keras.utils.plot_model(model, 'mnist_model.png')
#tf.keras.utils.plot_model(model, 'model_info.png', show_shape = True)
'''

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000,784).astype('float32')/255
model.compile(optimizer = tf.keras.optimizers.RMSprop(),
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])
history = model.fit(x_train, y_train, batch_size = 64, epochs = 5, validation_split = 0.2)
test_scores = model.evaluate(x_test, y_test, verbose = 0)
print('test loss:', test_scores[0])
print('test acc:', test_scores[1])

encode_input = tf.keras.Input(shape = (28, 28, 1), name = 'img')

h1 = layers.Conv2D(16,3,activation = 'relu')(encode_input)
h1 = layers.Conv2D(32,3,activation = 'relu')(h1)
h1 = layers.MaxPool2D(3)(h1)
h1 = layers.Conv2D(32,3, activation = 'relu')(h1)
h1 = layers.Conv2D(16,3, activation = 'relu')(h1)
encode_output = layers.GlobalMaxPool2D()(h1)

encode_model = tf.keras.Model(inputs = encode_input, outputs = encode_output, name = 'encoder')
print(encode_model.summary())


h2 = layers.Reshape((4,4,1))(encode_output)
h2 = layers.Conv2DTranspose(16, 3, activation = 'relu')(h2)
h2 = layers.Conv2DTranspose(32, 3, activation = 'relu')(h2)
h2 = layers.UpSampling2D(3)(h2)
h2 = layers.Conv2DTranspose(16,3,activation = 'relu')(h2)
decode_output = layers.Conv2DTranspose(1,3,activation = 'relu')(h2)

autoencoder = tf.keras.Model(inputs = encode_input, outputs = decode_output, name = 'autoencoder')
print(autoencoder.summary())

'''




encode_input = tf.keras.Input(shape= (28, 28, 1), name = 'img')

h1 = layers.Conv2D(16, 3, activation='relu')(encode_input)
h1 = layers.Conv2D(32, 3, activation='relu')(h1)
h1 = layers.MaxPool2D(3)(h1)
h1 = layers.Conv2D(32, 3, activation='relu')(h1)
h1 = layers.Conv2D(16, 3, activation='relu')(h1)
encode_output = layers.GlobalMaxPool2D()(h1)

encode_model = tf.keras.Model(
    inputs=encode_input, outputs=encode_output, name='encoder')
print(encode_model.summary())

decode_input = tf.keras.Input(shape = (16,),name = 'encoded_img')
h2 = layers.Reshape((4, 4, 1))(decode_input)
h2 = layers.Conv2DTranspose(16, 3, activation='relu')(h2)
h2 = layers.Conv2DTranspose(32, 3, activation='relu')(h2)
h2 = layers.UpSampling2D(3)(h2)
h2 = layers.Conv2DTranspose(16, 3, activation='relu')(h2)
decode_output = layers.Conv2DTranspose(1, 3, activation='relu')(h2)

decode_model = tf.keras.Model(inputs = decode_input, outputs = decode_output, name = 'deconde')
decode_model.summary()

autoecode_input = tf.keras.Input(shape = (28,28,1), name = 'img')
h3 = encode_model(autoecode_input)
autoencode_output = decode_model(h3)


autoencoder = tf.keras.Model(
    inputs=autoecode_input, outputs=autoencode_output, name='autoencoder')
print(autoencoder.summary())

'''
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255
autoencoder.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = autoencoder.fit(x_train, y_train, batch_size=64,
                    epochs=5, validation_split=0.2)
test_scores = autoencoder.evaluate(x_test, y_test, verbose=0)
'''

'''

num_words = 2000
num_tags = 12
num_departments = 4

body_input = tf.keras.Input(shape = (None,), name = 'body')
title_input = tf.keras.Input(shape = (None,), name = 'title')
tag_input = tf.keras.Input(shape = (num_tags,), name = 'tag')

body_feat = layers.Embedding(num_words, 64)(body_input)
title_feat = layers.Embedding(num_words, 64)(title_input)

body_feat = layers.LSTM(32)(body_feat)
title_feat = layers.LSTM(128)(title_feat)
features = layers.concatenate([title_feat, body_feat, tag_input])

priority_pred = layers.Dense(1, activation = 'sigmoid', name = 'priority')(features)
department_pred = layers.Dense(num_departments, activation = 'softmax', name = 'department')(features)

model = tf.keras.Model(inputs = [body_input, title_input, tag_input], outputs = [priority_pred, department_pred])

print(model.summary())

model.compile(optimizer = tf.keras.optimizers.RMSprop(1e-3),loss = {'priority': 'binary_crossentropy', 'department': 'categorical_crossentropy'}, loss_weights = [1., 0.2])

import numpy as np

title_data = np.random.randint(num_words, size = (1280,10))
body_data = np.random.randint(num_words, size = (1280,100))
tag_data = np.random.randint(2, size = (1280, num_tags)).astype('float32')

priority_label = np.random.random(size = (1280,1))
department_label = np.random.randint(2, size = (1280, num_departments))

history = model.fit(
    {'title': title_data, 'body': body_data, 'tag': tag_data},
    {'priority': priority_label, 'department': department_label},
    batch_size = 32,
    epochs = 5
)

'''

inputs = tf.keras.Input(shape = (32,32,3), name = 'img')
h1 = layers.Conv2D(32, 3, activation = 'relu')(inputs)
h1 = layers.Conv2D(64, 3, activation = 'relu')(h1)
block1_out = layers.MaxPool2D(3)(h1)

h2 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(block1_out)
h2 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(h2)
block2_out = layers.add([h2, block1_out])

h3 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(block2_out)
h3 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(h3)
block3_out = layers.add([h3, block2_out])

h4 = layers.Conv2D(64, 3, activation = 'relu')(block3_out)
h4 = layers.GlobalMaxPool2D()(h4)
h4 = layers.Dense(256, activation = 'relu')(h4)
h4 = layers.Dropout(0.5)(h4)
outputs = layers.Dense(10, activation = 'softmax')(h4)

model = tf.keras.Model(inputs, outputs, name = 'samll resnet')
print(model.summary())


(x_train, y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
# tf.keras.utils.to_categprical tansorframe to one_hot code
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model.compile(optimizer = tf.keras.optimizers.RMSprop(1e-3),
             loss = 'categorical_crossentropy',
             metrics = ['acc']
             )
'''
model.fit(
    x_train, y_train,
    batch_size = 64,
    epochs = 3,
    validation_split = 0.2
)

'''
###########################

from tensorflow.keras.applications import VGG16
vgg16 = VGG16()

feature_list = [layer.output for layer in vgg16.layers]
feat_ext_model = tf.keras.Model(inputs = vgg16.input, outputs = feature_list)

img = np.random.randn(1, 224, 224, 3).astype('float32')
#img = np.random.random((1,224,224,3).astype('float32'))


print(feat_ext_model.summary())
print(feat_ext_model(img))