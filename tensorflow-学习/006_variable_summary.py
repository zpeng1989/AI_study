import tensorflow as tf
my_var = tf.Variable(tf.ones([2,3]))

print(my_var)

try:
    with tf.device('/device:GPU:0'):
        v = tf.Variable(tf.zeros([10, 10]))
        print(v)
except:
    print('no gpu')
a = tf.Variable(1.0)
b = (a + 1) * 3

print(b)

a = tf.Variable(1.0)
b = (a.assign_add(2)) * 3

print(b)
