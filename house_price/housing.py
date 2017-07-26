import tensorflow as tf
import numpy as np

# Model linear regression y = Wx + b
x = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))
product = tf.matmul(x,W)
y = product+b
y_ = tf.placeholder(tf.float32, [None, 1])

# Cost function 1/n * sum((y_-y)**2)
cost = tf.reduce_mean(tf.square(y_-y))

# Training using Gradient Descent
train_step = tf.train.GradientDescentOptimizer(0.0000001)
