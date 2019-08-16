from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np

layer = keras.layers
print("tensorflow version check : ", tf.__version__)
print("gpu check", tf.test.is_gpu_available())


# 1. get dataset on memory
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("mnist dataset on memory")
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
print("The shape of train dataset : ", x_train.shape)
print("The shape of test dataset : ", x_test.shape)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(5000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(5000).batch(32)

inputs = keras.Input(shape=(28, 28, 1))
feature = layer.Conv2D(32, 3,activation='relu')(inputs)
feature = layer.MaxPool2D(pool_size=(2, 2))(feature)
feature = layer.Conv2D(64, 3, activation='relu')(feature)
feature = layer.MaxPool2D(pool_size=(2, 2))(feature)
flatten = layer.Flatten()(feature)
embedding = layer.Dense(128, activation='relu')(flatten)
scores = layer.Dense(10, activation='softmax')(embedding)
model = keras.Model(inputs, scores)

model.summary()

model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

print('========================Training===============================')

model.fit(x_train, y_train, epochs=5)

print('========================Evaluation===============================')

test_loss, test_acc = model.evaluate(test_ds)

del model
keras.backend.clear_session()
