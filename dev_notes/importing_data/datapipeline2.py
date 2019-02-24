import tensorflow as tf

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))

print(dataset1.output_types)
print(dataset1.output_shapes)
print(tf.random_uniform([4, 10]).shape)

https://www.tensorflow.org/guide/datasets