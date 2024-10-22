# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.keras.backend.clear_session()
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class MyLayer(layers.Layer):
    def __init__(self, input_dim = 32, unit = 32):
        super(MyLayer, self).__init__()
        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(initial_value = w_init(shape = (input_dim, unit), dtype = tf.float32), trainable = True)
        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value = b_init(shape = (unit,), dtype = tf.float32), trainable = True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias
'''
x = tf.ones((3,5))
my_layer = MyLayer(5,4)
out = my_layer(x)
'''
class MyLayer(layers.Layer):
    def __init__(self, input_dim = 32, unit = 32):
        super(MyLayer, self).__init__()
        self.weight = self.add_weight(shape= (input_dim, unit), initializer = keras.initializers.RandomNormal(), trainable = True)
        self.bias = self.add_weight(shape = (unit,), initializer = keras.initializers.Zeros(), trainable = True)
    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias
'''
x = tf.ones((3,5))
my_layer = MyLayer(5,4)
out = my_layer(x)
'''

##########

class AddLayer(layers.Layer):
    def __init__(self, input_dim = 32):
        super(AddLayer, self).__init__()
        self.sum = self.add_weight(shape = (input_dim,), initializer = keras.initializers.Zeros(), trainable = False)
    def call(self, inputs):
        self.sum.assign_add(tf.reduce_sum(inputs, axis = 0))
        return self.sum

x = tf.ones((3,3))
#print(x)
my_layer = AddLayer(3)
out = my_layer(x)
'''
print(out.numpy())
print(my_layer.weights)
print(my_layer.non_trainable_weights)
print(my_layer.trainable_weights)
'''


class MyLayer(layers.Layer):
    def __init__(self, unit = 32):
        super(MyLayer, self).__init__()
        self.unit = unit
    def build(self, input_shape):
        self.weight = self.add_weight(
            shape = (input_shape[-1], self.unit),
            initializer = keras.initializers.RandomNormal(),
            trainable = True
        )
        self.bias = self.add_weight(
            shape = (self.unit,),
            initializer = keras.initializers.Zeros(),
            trainable = True
        )
    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias

'''
my_layer = MyLayer(3)
x = tf.ones((3,5))
out = my_layer(x)
print(x)
print(out)

my_layer = MyLayer(3)
x = tf.ones((2, 2))
out = my_layer(x)
print(x)
print(out)
'''


class MyBlock(layers.Layer):
    def __init__(self):
        super(MyBlock, self).__init__()
        self.layer1 = MyLayer(32)
        self.layer2 = MyLayer(16)
        self.layer3 = MyLayer(2)
    def call(self, inputs):
        h1 = self.layer1(inputs)
        h1 = tf.nn.relu(h1)
        h2 = self.layer2(h1)
        h2 = tf.nn.relu(h2)
        return self.layer3(h2)

my_block = MyBlock()
print('trainable weights:', len(my_block. trainable_weights))
y = my_block(tf.ones(shape = (3, 64)))
print('trainable weights:', len(my_block. trainable_weights))


class LossLayer(layers.Layer):
    def __init__(self, rate = 1e-2):
        super(LossLayer, self).__init__()
        self.rate = rate
    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs

class OutLayers(layers.Layer):
    def __init__(self):
        super(OutLayers, self).__init__()
        self.loss_fun = LossLayer(1e-2)
    def call(self, inputs):
        return self.loss_fun(inputs)

my_layer = OutLayers()
print(len(my_layer.losses))
y = my_layer(tf.zeros(1,1))
print(len(my_layer.losses))
y = my_layer(tf.zeros(1,1))
print(len(my_layer.losses))


class OuterLayer(layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.dense = layers.Dense(32, kernel_regularizer = tf.keras.regularizers.l2(1e-3))
    def call(self, inputs):
        return self.dense(inputs)

my_layer = OuterLayer()
y = my_layer(tf.zeros((1,1)))
print(my_layer.losses)
print(my_layer.weights)



class Linear(layers.Layer):
    def __init__(self, units = 32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units
    def build(self, input_shape):
        self.w = self.add_weight(shape = (input_shape[-1], self.units),
                                 initializer = 'random_normal',
                                 trainable = True
                                )
        self.b = self.add_weight(shape = (self.units,),
                                 initializer = 'random_normal',
                                 trainable = True
                                )
    def call(self, inputs):
        return tf.matmal(inputs, self.w) + self.b
    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config

layer = Linear(125)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape = (batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(layers.Layer):
    def __init__(self, latent_dim = 32, intermediate_dim = 64, name = 'encoder', **kwargs):
        super(Encoder, self).__init__(name = name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation = 'relu')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        z_mean = self.dense_mean(h1)
        z_log_var = self.dense_log_var(h1)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

class Encoder(layers.Layer):
    def __init__(self, latent_dim=32, intermediate_dim = 64, name = 'encoder', **kwargs):
        super(Encoder, self).__init__(name = name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation = 'relu')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
        
    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        z_mean = self.dense_mean(h1)
        z_log_var = self.dense_log_var(h1)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    def __init__(self, original_dim, intermediate_dim = 64, name = 'decoder', **kwargs):
        super(Decoder, self).__init__(name = name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation = 'relu')
        self.dense_output = layers.Dense(original_dim, activation = 'sigmoid')
    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        return self.dense_output(h1)


class Decoder(layers.Layer):
    def __init__(self, original_dim, intermediate_dim = 64, name = 'decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = layers.Dense(original_dim, activation='sigmoid')
    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        return self.dense_output(h1)
 

class VAE(tf.keras.Model):
    def __init__(self, original_dim, latent_dim = 32, intermediate_dim = 64, name = 'encoder', **kwargs):
        super(VAE, self).__init__(name = name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim = latent_dim, intermediate_dim = intermediate_dim)
        self.decoder = Decoder(original_dim = original_dim, intermediate_dim = intermediate_dim)
    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5*tf.reduce_sum(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed

class VAE(tf.keras.Model):
    def __init__(self, original_dim, latent_dim=32, intermediate_dim=64, name='encoder', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim = latent_dim, intermediate_dim = intermediate_dim)
        self.decoder = Decoder(original_dim = original_dim, intermediate_dim = intermediate_dim)
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5*tf.reduce_sum(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed



(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
vae = VAE(784,32,64)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
'''
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)

'''
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(64)

original_dim = 784
vae = VAE(original_dim, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

for epoch in range(3):
    print('Start of epochs %d' %(epoch, ))
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses)
        grads = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(grads, vae.trainable_variables))
        loss_metric(loss)

        if step % 100 == 0:
            print('step %s: mean loss = %s' %(step, loss_metric.result()))

