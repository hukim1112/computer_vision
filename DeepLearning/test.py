import tensorflow as tf

a = tf.Variable(1.0)
b = tf.Variable(2.0)

@tf.function
def f(x, y):
  a.assign(y * b) # 2*2
  b.assign_add(x * a) #2+1*4
  return a + b

print(f(1,2))
