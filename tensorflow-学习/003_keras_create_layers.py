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





