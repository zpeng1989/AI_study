import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tensorflow.keras.layers as layers

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top','Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)


train_images = train_images / 255.0
test_images = test_images / 255.0


model = keras.Sequential(
    [
        layers.Flatten(input_shape = [28,28]),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(10,activation = 'softmax')
    ]
)

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())
model.fit(train_images, train_labels, epochs = 5)

predictions = model.predict(test_images)

print(predictions)
