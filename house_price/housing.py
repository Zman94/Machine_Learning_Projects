import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

steps = 1000

### Fake housing prices
xs = [[j for i in range(1)] for j in range(100)]
ys = [[2*j for i in range(1)] for j in range(100)]

### Model linear regression y = Wx + b ###
x = tf.placeholder(tf.float32, [None, 1])
# x = tf.reduce_mean(x)
W = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))
h = tf.matmul(x,W) + b
y = tf.placeholder(tf.float32, [None, 1])

### Cost function 1/(m) * sum((y_-y)**2) ###
cost = tf.reduce_mean(tf.square(h-y))

### Training using Gradient Descent ###
optimizer = tf.train.GradientDescentOptimizer(0.0003)
train = optimizer.minimize(cost)

### Training Loop ###
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(steps):
    sess.run(train, {x:xs, y:ys})
    print("After %d iterationi:" % i)
    print("W: %f" % sess.run(W))
    print("b: %f" % sess.run(b))

### Results ###
# curr_W = sess.run(W)
# curr_b = sess.run(b)
curr_W, curr_b, curr_loss = sess.run([W, b, cost], feed_dict={x:xs, y:ys})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
