from __future__ import absolute_import, division, print_function, unicode_literals

# https://www.tensorflow.org/beta/tutorials/eager/tf_function

'''
In TensorFlow 2.0, eager execution is turned on by default. The user interface is intuitive and flexible (running one-off operations is much easier and faster), but this can come at the expense of performance and deployability.

To get peak performance and to make your model deployable anywhere, use tf.function to make graphs out of your programs. Thanks to AutoGraph, a surprising amount of Python code just works with tf.function, but there are still pitfalls to be wary of.

The main takeaways and recommendations are:

1.Don't rely on Python side effects like object mutation or list appends.
2.tf.function works best with TensorFlow ops, rather than NumPy ops or Python primitives.
3.When in doubt, use the for x in y idiom.

Why we use @tf.Functions
1. To ask tensorflow to retrace graphs for dynamic input type like python-style.
2. To get the same results between eager excution and graph-flow.
3. To use if statement and for statement properly in graph-flow too.


'''

try:
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf



# Functions have gradients

@tf.function
def add(a, b):
  return a + b

v = tf.Variable(1.0)
with tf.GradientTape() as tape:
  result = add(v, 1.0)
print('1.gradient', tape.gradient(result, v))
