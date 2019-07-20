# -*- coding = utf-8 -*-
# https://www.tensorflow.org/api_docs/python/tf


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import pandas as pd


#layer = tf.keras.layers.Dense(100)
mylayer = tf.keras.layers.Dense(100, input_shape = (None, 3))


#print(tf.ones([10,6]))
#print(mylayer(tf.ones([10,6])))

#print(mylayer.variables)
#print(mylayer.kernel, mylayer.bias)

class ResnetBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetBlock, self).__init__(name = 'resnet_block')
        filter1, filter2, filter3 = filters
        self.conv1 = tf.keras.layers.Conv2D(filter1, (1, 1))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filter2, kernel_size, padding = 'same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filter3, (1,1))
        self.bn3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training = False):
        x = self.conv1(inputs)
        x = self.bn1(x, training = training)
        x = self.conv2(x)
        x = self.bn2(x, training = training)
        x = self.conv3(x)
        x = self.bn3(x, training = training)
        
        x += inputs
        outputs = tf.nn.relu(x)
        return outputs

resnetBlock = ResnetBlock(2, [6, 4, 9])
print('++++++++++++++++++++++++++++++++++++++First Out++++++++++++++++++++++++++++++++++++')
print(tf.ones([1, 3, 9, 9]))
print(resnetBlock(tf.ones([1, 3, 9, 9])))

print('++++++++++++++++++++++++++++++++++++++Second Out++++++++++++++++++++++++++++++++++++')

print([x.name for x in resnetBlock.trainable_variables])



seq_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(1, 1, input_shape = (None, None, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(2, 1, padding = 'same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(3, 1),
    tf.keras.layers.BatchNormalization()
    ])


print(tf.ones([1,2,5,3]))
print(seq_model(tf.ones([1,2,5,3])))




