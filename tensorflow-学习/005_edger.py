import numpy as np
import tensorflow as tf

x = [[3.,4.]]
y = [[3.],[4.]]
m = tf.matmul(y,x)
#print(m.numpy())

a = tf.constant([[1,9],[3,6]])
#print(a+m.numpy())

b = tf.add(a, 2)
#print(b)
#print(a*b)

s = np.multiply(a,b)
#print(s)

def fizzbuzz(max_num):
    counter = tf.constant(0)
    max_num = tf.convert_to_tensor(max_num)
    for num in range(1, max_num.numpy() + 1):
        num = tf.constant(num)
        if int(num % 3) == 0 and int(num % 5) == 0:
            print('FizzBuzz')
        if int(num % 3) == 0:
            print('Fizz')
        if int(num % 5) == 0:
            print('Buzz')
        else:
            print(num.numpy())
        counter += 1
#fizzbuzz(100)

class MySimpleLayer(tf.keras.layers.Layer):
    def __init__(self, output_units):
        super(MysimpleLayer, self).__init__
        self.output_units = output_units
        self.dynamic = True
    def build(self, input_shape):
        self.kernal = self.add_variable('kernal', [input_shape[-1], self.output_units])
    def call(self, input):
        return tf.matmul(input, self.kernel)

class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units = 10)
        self.dense2 = tf.keras.layers.Dense(units = 10)

    def call(self, inputs):
        result = self.dense1(inputs)
        result = self.dense2(result)
        result = self.dense2(result)
        return result

model = MNISTModel()


    

