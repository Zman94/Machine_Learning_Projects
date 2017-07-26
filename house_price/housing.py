import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

### Model linear regression y = Wx + b ###
# x = tf.placeholder(tf.float32, [None, 1])
x = tf.placeholder(tf.float32)
# x = tf.reduce_mean(x)
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# W = tf.Variable(tf.zeros([1,1]))
# b = tf.Variable(tf.zeros([1]))
h = W*x +b
# y_ = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32)

### Cost function 1/(2m) * sum((y_-y)**2) ###
cost = tf.reduce_mean(tf.square(h-y))

### Training using Gradient Descent ###
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)

### Training data ###
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

### Training Loop ###
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

### Results ###
curr_W, curr_b, curr_loss = sess.run([W, b, cost], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
