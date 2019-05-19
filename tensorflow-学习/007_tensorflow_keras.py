from __future__ import print_function
import tensorflow as tf
import tensorflow.keras as keras

@tf.function
def simple_nn_layer(x, y):
    return tf.nn.relu(tf.matmul(x,y))

x = tf.random.uniform((3,3))
y = tf.random.uniform((3,3))

print(simple_nn_layer(x,y))

print(simple_nn_layer)


def linear_layer(x):
    return 2 * x + 1

@tf.function
def deep_net(x):
    return tf.nn.relu(linear_layer(x))


print('++++++++++++test++++++++++++++')
print(deep_net(tf.constant((1, 2, 3))))

@tf.function
def square_if_positive(x):
    if x > 0:
        x = x * x
    else:
        x = 0
    return x

print('square_if_positive(2) = {}'.format(square_if_positive(2)))
print('square_if_positive(-2) = {}'.format(square_if_positive(-2)))


@tf.function
def sum_even(items):
    s = 0
    for c in items:
        if c % 2 > 0:
            continue
        s += c
    return s

print(sum_even(tf.constant([10,12,15,20])))

print(tf.autograph.to_code(sum_even.python_function, experimental_optional_features=None))
#from __future__ import print_function
'''
def tf_sum_even(item):
    do_return = False
    retval = None
    s = 0

    def loop_body(loop_vars, s_2):
        c = loop_vars
        continue_ = False
        cond = c % 2 > 0

        def if_true():
            continue_ = True
            return continue_
        def if_false():
            return continue_
        continue_ = ag__.if_stmt(cond, it_true, if_false)
        cond_1 = ag__.not_(continue_)

        def if_true_1():
            s_1, = s_2,
            s_1 += c
            return s_1
        def if_flase_1():
            return s_2
        s_2 = ag__.if_stmt(cond_1, if_true_1, if_false_1)
        return s_2,
    s, = ag__.for_stmt(items, None, loop_body, (s,))
    do_return = True
    retval_ = s
    return retval_

'''

@tf.function
def fizzbuzz(n):
    msg = tf.constant('')
    for i in tf.range(n):
        if tf.equal(i % 3, 0):
            msg += 'Fizz'
        elif tf.equal(i % 5, 0):
            msg += 'Buzz'
        else:
            msg += tf.as_string(i)
        msg += '\n'
    return msg

print(fizzbuzz(tf.constant(15)).numpy().decode())


class CustomModel(tf.keras.models.Model):
    @tf.function
    def call(self, input_data):
        if tf.reduce_mean(input_data) > 0:
            return input_data
        else:
            return input_data // 2
model = CustomModel()

print(model(tf.constant([-1, -4])))


v = tf.Variable(5)
print(v)

@tf.function
def find_next_odd():
    v.assign(v+1)
    if tf.equal(v % 2, 0):
        v.assign(v + 1)

print(find_next_odd())
print(v)

def prepare_mnist_features_and_labels(x,y):
    x = tf.cast(x, tf.float32)/255
    y = tf.cast(y, tf.int64)
    return x,y
'''

def prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y

'''


def mnist_dataset():
    (x,y), _ = tf.keras.datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x,y))
    ds = ds.map(prepare_mnist_features_and_labels)
    #ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds
'''

def mnist_dataset():
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds
'''

train_dataset = mnist_dataset()
model = tf.keras.Sequential((
    tf.keras.layers.Reshape(target_shape = (28 * 28,), input_shape = (28, 28)),
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(100, activation = 'relu'),
    tf.keras.layers.Dense(10)))

model.build()
print(model.summary())
optimizer = tf.keras.optimizers.Adam()
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

'''

train_dataset = mnist_dataset()
model = tf.keras.Sequential((
    tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10)))
model.build()
optimizer = tf.keras.optimizers.Adam()
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

'''

def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(y, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    compute_accuracy(y, logits)
    return loss


'''
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(y, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    compute_accuracy(y, logits)
    return loss
'''


@tf.function
def train(model, optimizer):
    train_ds = mnist_dataset()
    step = 0
    loss = 0.0
    accuracy = 0.0
    for x, y in train_ds:
        step += 1
        loss = train_one_step(model, optimizer, x, y)
        if tf.equal(step % 10, 0):
            tf.print('Step', step, ': loss', loss, '; accuracy',compute_accuracy.result())
    return step, loss, accuracy
    
step, loss, accuracy = train(model, optimizer)
print('Final step', step,':loss', loss, '; accuracy', compute_accuracy.result())



'''
@tf.function
def train(model, optimizer):
    train_ds = mnist_dataset()
    step = 0
    loss = 0.0
    accuracy = 0.0
    for x, y in train_ds:
        step += 1
        loss = train_one_step(model, optimizer, x, y)
        if tf.equal(step % 10, 0):
            tf.print('Step', step, ': loss', loss,
               '; accuracy', compute_accuracy.result())
    return step, loss, accuracy


step, loss, accuracy = train(model, optimizer)
print('Final step', step, ': loss', loss,
      '; accuracy', compute_accuracy.result())

'''