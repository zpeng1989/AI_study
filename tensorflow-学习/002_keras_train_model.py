import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,), name='mnist_input')
h1 = layers.Dense(64, activation='relu')(inputs)
h1 = layers.Dense(64, activation='relu')(h1)
outputs = layers.Dense(10, activation='softmax')(h1)
model = keras.Model(inputs, outputs)
# keras.utils.plot_model(model, 'net001.png', show_shapes=True)

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 训练模型
#history = model.fit(x_train, y_train, batch_size=64, epochs=3,
#                    validation_data=(x_val, y_val))
print('history:')
#print(history.history)

#result = model.evaluate(x_test, y_test, batch_size=128)
print('evaluate:')
#print(result)
#pred = model.predict(x_test[:2])
print('predict:')
#print(pred)

class CatgoricalTruePostives(keras.metrics.Metric):
    def __init__(self, name = 'binary_true_postives', **kwargs):
        super(CatgoricalTruePostives, self).__init__(name = name, **kwargs)
        self.true_postives = self.add_weight(name = 'tp', initializer = 'zeros')
    def update_state(self, y_true, y_pred, sample_weight = None):
        y_pred = tf.argmax(y_pred)
        y_true = tf.equal(tf.cast(y_pred, tf.int32), tf.cast(y_true, tf.int32))
        y_true = tf.cast(y_true, tf.float32)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            y_true = tf.multiply(sample_weight, y_true)
        return self.true_postives.assign_add(tf.reduce_sum(y_true))
    def result(self):
        return tf.identity(self.true_postives)
    def reset_states(self):
        self.true_postives.assign(0.)

model.compile(optimizer = keras.optimizers.RMSprop(1e-3), 
              loss = keras.losses.SparseCategoricalCrossentropy(), 
              metrics = [CatgoricalTruePostives()])

#model.fit(x_train, y_train, batch_size = 64, epochs =3)


class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs)*0.1)
        return inputs

inputs = keras.Input(shape = (784,), name = 'mnist_input')
h1 = layers.Dense(64, activation = 'relu')(inputs)
h1 = ActivityRegularizationLayer()(h1)
h1 = layers.Dense(64, activation = 'relu')(h1)
outputs = layers.Dense(10,activation = 'softmax')(h1)
model = keras.Model(inputs, outputs)
print(model.summary())

model.compile(optimizer = keras.optimizers.RMSprop(),
            loss = keras.losses.SparseCategoricalCrossentropy(),
            metrics = [keras.metrics.SparseCategoricalAccuracy()]
            )

#model.fit(x_train,y_train, batch_size = 32, epochs =3)


class MetricLoggingLayer(layers.Layer):
    def call(self, inputs):
        self.add_metric(keras.backend.std(inputs),
                        name = 'std_of_activation',
                        aggregation = 'mean'
                        )
        return inputs

inputs = keras.Input(shape = (784,), name = 'mnist_input')
h1 = layers.Dense(64, activation = 'relu')(inputs)
h2 = layers.Dense(64, activation = 'relu')(h1)
outputs = layers.Dense(10, activation = 'softmax')(h2)
model = keras.Model(inputs, outputs)
print(model.summary())

model.add_metric(keras.backend.std(inputs),name = 'std_of_activation', aggregation = 'mean')

model.add_loss(tf.reduce_sum(h1)*0.1)

print(model.summary())
model.compile(optimizer = keras.optimizers.RMSprop(),
            loss = keras.losses.SparseCategoricalCrossentropy(),
            metrics = [keras.metrics.SparseCategoricalAccuracy()])

#model.fit(x_train, y_train, batch_size = 32, epochs = 3)


def get_compiled_model():
    inputs = keras.Input(shape=(784,), name='mnist_input')
    h1 = layers.Dense(64, activation='relu')(inputs)
    h2 = layers.Dense(64, activation='relu')(h1)
    outputs = layers.Dense(10, activation='softmax')(h2)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model


model = get_compiled_model()
print(model.summary())

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(64)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

print(val_dataset)

#model.fit(train_dataset, epochs = 3, steps_per_epoch = 100, validation_data = val_dataset, validation_steps = 3)

import numpy as np
model = get_compiled_model()
class_weight = {i:1.0 for i in range(10)}
class_weight[5] = 2.0
print(class_weight)
#model.fit(x_train,y_train
#          ,class_weight = class_weight
#          ,batch_size = 64
#          ,epochs = 4)

model = get_compiled_model()
sample_weight = np.ones(shape = len(y_train))
sample_weight[y_train == 5] = 2.0
#model.fit(x_train,y_train,
#        sample_weight = sample_weight,
#        batch_size = 64,
#        epochs = 4)

model = get_compiled_model()
sample_weight = np.ones(shape = len(y_train))
print(sample_weight)
sample_weight[y_train == 5] = 2.0

train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train, sample_weight))
train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(64)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

#model.fit(train_dataset, epochs = 3)

images_input = keras.Input(shape = (32, 32, 3), name = 'img_input')
timerseries_input = keras.Input(shape = (None, 10), name = 'ts_input')


print(images_input)
print(timerseries_input)
x1 = layers.Conv2D(3,3)(images_input)
x1 = layers.GlobalMaxPooling2D()(x1)

x2 = layers.Conv1D(3,3)(timerseries_input)
x2 = layers.GlobalMaxPooling1D()(x2)

x = layers.concatenate([x1, x2])

score_output = layers.Dense(1, name = 'score_output')(x)
class_output = layers.Dense(5, activation = 'softmax', name = 'class_output')(x)

model = keras.Model(inputs = [images_input, timerseries_input],
                    outputs = [score_output, class_output]
                    )


model.compile(
    optimizer = keras.optimizers.RMSprop(1e-3),
    loss = {
        'score_output': keras.losses.MeanSquaredError(),
        'class_output': keras.losses.CategoricalCrossentropy()
    },
    metrics = {
        'score_output':[keras.metrics.MeanAbsolutePercentageError(),
                        keras.metrics.MeanAbsoluteError()],
        'class_output': [keras.metrics.CategoricalAccuracy()]
    },
    loss_weight = {'score_output':2, 'class_output':1}
)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[None, keras.losses.CategoricalCrossentropy()])

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={'class_output': keras.losses.CategoricalCrossentropy()})



model = get_compiled_model()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 1e-2,
        patience = 2,
        verbose = 1
    )
]

#model.fit(x_train, y_train, epochs = 20, batch_size = 64, callbacks = callbacks, validation_split = 0.2)

model= get_compiled_model()
check_callback = keras.callbacks.ModelCheckpoint(
    filepath = 'mymodel_{epoch}.h5',
    save_best_only = True,
    monitor = 'val_loss',
    verbose = 1
)

#model.fit(x_train, y_train,
#    epochs = 3,
#    batch_size = 64,
#    callbacks = [check_callback],
#    validation_split = 0.2
#)

initial_learning_rate = 0.1

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps = 10000,
    decay_rate = 0.96,
    staircase = True
)

optimizer = keras.optimizers.RMSprop(learning_rate = lr_schedule)

tensorboard_cbk = keras.callbacks.TensorBoard(
    log_dir='./full_path_to_your_logs')
#model.fit(x_train, y_train,
#          epochs=5,
#          batch_size=64,
#          callbacks=[tensorboard_cbk],
#          validation_split=0.2)


inputs = keras.Input(shape=(784,), name = 'digits')

x = layers.Dense(64, activation = 'relu', name = 'dense_1')(inputs)
x = layers.Dense(64, activation = 'relu', name = 'dense_2')(x)
outputs = layers.Dense(10, activation = 'softmax', name = 'predictions')(x)

model = keras.Model(inputs = inputs, outputs = outputs)

optimizer = keras.optimizers.SGD(learning_rate = 1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(batch_size)

'''
for epoch in range(3):
    print('epoch:', epoch)
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train)
            loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if step % 200 == 0:
            print('Training loss (for one batch) at step %s: %s'%(step, float(loss_value)))
            print('Seen so far: %s samples'%((step + 1)*64))

'''
inputs = keras.Input(shape = (784,), name = 'digits')
x = layers.Dense(64, activation = 'relu', name = 'dense_1')(inputs)
x = layers.Dense(64, activation = 'relu', name = 'dense_2')(x)
outputs = layers.Dense(10, activation = 'softmax', name = 'predictions')(x)
model = keras.Model(inputs = inputs, outputs = outputs)

optimizer = keras.optimizers.SGD(learning_rate = 1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size = 1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

'''
for epoch in range(3):
    print('Start of epoch %d' %(epoch,))
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_acc_metric(y_batch_train, logits)
        if step % 200 == 0:
            print('Training loss (for one batch) at step %s: %s' %
                  (step, float(loss_value)))
            print('Seen so far: %s samples' % ((step + 1)*64))
    train_acc = train_acc_metric.result()
    print('training acc over epoch %s'%(float(train_acc),))
    train_acc_metric.reset_states()
    for x_batch_val, y_batch_val in val_dataset: 
        val_logits = model(x_batch_val)   
        val_acc_metric(y_batch_val,val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print('Validation acc: %s'%(float(val_acc),))

'''

class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2*tf.reduce_sum(inputs))
        return inputs


inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
# Insert activity regularization as a layer
x = ActivityRegularizationLayer()(x)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
logits = model(x_train[:64])
print(model.losses)
logits = model(x_train[:64])
logits = model(x_train[64: 128])
logits = model(x_train[128: 192])
print(model.losses)

optimizer = keras.optimizers.SGD(learning_rate = 1e-3)

for epoch in range(3):
    print('Start of epoch %d' %(epoch,))
    for step ,(x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train)
            loss_value = loss_fn(y_batch_train, logits)
            loss_value += sum(model.losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 200 == 0:
            print('training loss (for one batch) at step %s : %s'%(step, float(loss_value)))
            print('Seen so far: %s samples'%((step + 1)*64))






