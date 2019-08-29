import tensorflow as tf

a = tf.Variable(1.0)
b = tf.Variable(2.0)

@tf.function
def f(x, y):
  a.assign(y * b)
  b.assign_add(x * a)
  return a + b

print(f(1,2))
