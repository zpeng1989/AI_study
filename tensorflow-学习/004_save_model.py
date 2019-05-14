from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.keras.backend.clear_session()
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

inputs = keras.Input(shape = (784,), name = 'digits')
x = layers.Dense(64, activation = 'relu', name = 'dense_1')(inputs)
x = layers.Dense(64, activation = 'relu', name = 'dense_2')(x)
outputs = layers.Dense(10, activation = 'softmax', name = 'predications')(x)

model = keras.Model(inputs = inputs, outputs = outputs, name = '3_layer_mlp')
print(model.summary())


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000,784).astype('float32')/255

model.compile(loss ='sparse_categorical_crossentropy', optimizer = keras.optimizers.RMSprop())

history = model.fit(x_train, y_train, batch_size = 64, epochs = 3)
predictions = model.predict(x_test)

import numpy as np

model.save('the_save_model.h5')
new_model = keras.models.load_model('the_save_model.h5')
new_prediction = new_model.predict(x_test)
print(np.testing.assert_allclose(predictions, new_prediction, atol = 1e-6))



keras.experimental.export_save_model(model, 'saved_model')
new_model = keras.experimental.load_form_saved_model('saved_model')
new_prediction = new_model.predict(x_test)
np.testing.assert_allclose(predictions, new_prediction, atol = 1e-6)


config = model.get_config()
reinitalized_model = keras.Model.from_config(config)


