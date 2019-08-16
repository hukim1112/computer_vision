from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
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

# 3. Build the tf.keras model using the Keras model subclassing API:

class MyModel(Model): # inherit Model class of the tf.keras
  def __init__(self, num_classes):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.dropout = tf.keras.layers.Dropout(0.5)
    self.d2 = Dense(num_classes, activation='softmax')

  def call(self, x, training=False):
    x = self.conv1(x)
    x = self.flatten(x)
    if training:
      x = self.dropout(x, training=training)
    x = self.d1(x)
    return self.d2(x)

# 4. model training
# Create an instance of the model
model = MyModel(num_classes=10)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# bce = tf.keras.losses.BinaryCrossentropy()
# loss = bce([0., 0., 1., 1.], [1., 1., 1., 0.]) # tensorflow.python.framework.ops.EagerTensor
# print('Loss: ', loss.numpy())  # Loss: 11.522857
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

  # Reset the metrics for the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()


# 5. Save and Load your model

# Save weights to a TensorFlow Checkpoint file
model.save_weights('./weights/my_model')

# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('./weights/my_model')
del model
keras.backend.clear_session()
