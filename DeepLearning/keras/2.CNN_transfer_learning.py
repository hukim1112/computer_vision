from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
#https://github.com/keras-team/keras/issues/6977
layer = tf.keras.layers
# print("tensorflow version check : ", tf.__version__)
# print("gpu check : ", tf.test.is_gpu_available())

# 1. get dataset on memory
cifar100 = tf.keras.datasets.cifar100
train, test = cifar100.load_data(label_mode='fine')
train_ds = tf.data.Dataset.from_tensor_slices((train[0], train[1])).shuffle(5000).batch(64)
test_ds = tf.data.Dataset.from_tensor_slices((test[0], test[1])).shuffle(5000).batch(64)
def parse_image(image, label):
  image = tf.image.resize(image, [128, 128])
  return image, label
train_ds = train_ds.map(parse_image)
test_ds = test_ds.map(parse_image)
IMG_SIZE=128
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# 2. Create the base model from the pre-trained convnets

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
# 2.1 Freeze the convolutional base and use it as feature extractor
base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))
# Fine tune from this layer onwards
fine_tune_at = 145
# Freeze all the layers before the `fine_tune_at` layer

for l in base_model.layers[:fine_tune_at]:
    l.trainable = False

global_average_layer = layer.GlobalAveragePooling2D()
prediction_layer = layer.Dense(100, activation='softmax')

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])


# inputs = keras.Input(shape=(128, 128, 3))
# feature_batch = base_model(inputs)
# global_average_layer = layer.Flatten()(feature_batch)
# prediction = layer.Dense(100, activation='softmax')(global_average_layer)
# model = keras.Model(inputs, prediction)
# model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

print(len(model.trainable_variables))

print('========================Training===============================')

model.fit(train_ds, epochs=5, validation_data=train_ds)

# print('========================Evaluation===============================')
#
# test_loss, test_acc = model.evaluate(test_ds)

del model
keras.backend.clear_session()
