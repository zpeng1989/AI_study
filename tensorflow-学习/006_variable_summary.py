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

class MyModuleOne(tf.Module):
    def __init__(self):
        self.v0 = tf.Variable(1.0)
        self.vs = [tf.Variable(x) for x in range(10)]
class MyOtherModule(tf.Module):
    def __init__(self):
        self.m = MyModuleOne()
        self.v = tf.Variable(10.0)

m = MyOtherModule()
print(m.variables)
len(m.variables)




