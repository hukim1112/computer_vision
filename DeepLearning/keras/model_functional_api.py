from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

print("tensorflow version check : ", tf.__version__)
tf.test.is_gpu_available()

# 1. get dataset on memory
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("mnist dataset on memory")
x_train, x_test = x_train / 255.0, x_test / 255.0
print("The shape of train dataset : ", x_train.shape)
print("The shape of test dataset : ", x_test.shape)

# Add a channels dimension
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# 2. Use tf.data to define a data-pipeline
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32).repeat()

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

inputs = tf.keras.Input(shape=(28, 28), batch_size=None, name=None, dtype=None, sparse=False, tensor=None)

