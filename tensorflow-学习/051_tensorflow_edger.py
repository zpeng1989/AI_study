from __fture__ import absolute_import, division, print_function

import tensorflow as tf

print(tf.add(1,2))
print(tf.add([3.8],[2,5]))

print(tf.square(6))
print(tf.reduce_sum([7,8,9]))
print(tf.square(3) + tf.square(4))


#tf.Tensor(3, shape = (), stype = int32)
#tf.Tensor([5 13], shape = (2,),dtype = int32)

#tf.Tensor(36, shape = (), dtype = int32)



x = tf.matmul([[3],[6]],[[2]])
print(x)
print(x.shape)
print(x.dtype)