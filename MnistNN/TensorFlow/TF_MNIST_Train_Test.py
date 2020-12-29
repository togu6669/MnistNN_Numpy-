# TensorFlow version of MNIST recognizer

# MNIST resources
# https://www.python-course.eu/neural_network_mnist.php

import numpy as np
import tensorflow as tf
from timeit import default_timer as timer

# Read training images
# labels are being read in the format (60000,) - [5, 0, 4, ..., 5, 6, 8], dtype=uint8)

mnist = tf.keras.datasets.mnist

(images, labels), (test_images, test_labels) = mnist.load_data()
images, test_images = images / 255.0, test_images / 255.0


start = timer()
# labels.shape
# initialize network layer: No of Neurons, Previous Layer, Bias, Learning Rate, Activation Function
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'), # 256 give accuracy above 99%
  tf.keras.layers.Dropout(0.2), # does not help
  tf.keras.layers.Dense(10)
])

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(500, activation='sigmoid'),
#   tf.keras.layers.Dense(400, activation='sigmoid'),
#   tf.keras.layers.Dense(10, activation='sigmoid')
# ])

# CHECK model
# predictions = model(images[:1]).numpy()
# predictions

# tf.nn.sigmoid(predictions).numpy()
# tf.nn.softmax(predictions).numpy()
# tf.nn.softplus(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# CHECK loss fuction 
# loss_fn(labels[:1], predictions).numpy()

# put all together in the model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
# model.compile(optimizer='SGD',
#               loss=loss_fn,
#               metrics=['accuracy'])

model.fit(images, labels, epochs=5)

model.evaluate(images, labels, verbose=2)

end = timer()
sectime = end - start
mintime = sectime / 60
print(' Time of training : ', sectime, ' sec, ', mintime, ' min')
