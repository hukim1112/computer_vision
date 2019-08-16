from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np

layer = tf.keras.layers
print("tensorflow version check : ", tf.__version__)
print("gpu check : ", tf.test.is_gpu_available())

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

item = iter(train_ds).next()

print(item[1].numpy())
