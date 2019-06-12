# -*- coding: utf-8 -*

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import time

(train_image, train_labels),(_,_) = tf.keras.datasets.mnist.load_data()
train_image = train_image.reshape(train_image.shape[0], 28, 28, 1).astype('float32')
train_image= (train_image- 127.5)/127.5

BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(train_image).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential()
    #model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias = False, input_shape = (100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7,7, 256)))
    assert model.output_shape == (None, 7, 7, 256)
    model.add(layers.Conv2DTranspose(128, (5,5), strides = (1,1), padding = 'same', use_bias = False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5,5), strides = (2,2), padding = 'same', use_bias = False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides = (2,2), padding = 'same', use_bias = False, activation = 'tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model


geneerator = make_generator_model()


print(geneerator.summary())
noise = tf.random.normal([1,100])
generated_image = geneerator(noise, training = False)



def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides = (2, 2), padding = 'same', input_shape = [28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5,5), strides = (2,2), padding = 'same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model


discriminator = make_discriminator_model()
decision = discriminator(generated_image)

print(decision)


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def geneerator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer= generator_optimizer, discriminator_optimizer = discriminator_optimizer, generator = geneerator, discriminator = discriminator)




EPOCHS = 50
nosie_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, nosie_dim])

@tf.function
def train_step(images):
    nosie = tf.random.normal([BATCH_SIZE, nosie_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = geneerator(nosie, training = True)
        #generated_images = generator(noise, training=True)
        real_output = discriminator(images, training = True)
        fake_output = discriminator(generated_images, training = True)
        gen_loss = geneerator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, geneerator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, geneerator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))    

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False. 
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    #fig = plt.figure(figsize=(4,4))

    #for i in range(predictions.shape[0]):
        #plt.subplot(4, 4, i+1)
        #plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        #plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)
        #display.clear_output(wait = True)
        generate_and_save_images(geneerator, epoch + 1, seed)
        if (epochs + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    #display.clear_output(wait=True)
    generate_and_save_images(geneerator, epochs, seed)

train(train_dataset, EPOCHS)




