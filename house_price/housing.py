import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

# Model linear regression y = Wx + b
# x = tf.placeholder(tf.float32, [None, 1])
x = tf.placeholder(tf.float32)
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# W = tf.Variable(tf.zeros([1,1]))
# b = tf.Variable(tf.zeros([1]))
# y = tf.matmul(x,W) + b
y = W*x +b
# y_ = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32)

# Cost function 1/(2m) * sum((y_-y)**2)
cost = tf.reduce_mean(tf.square(y-y_))/2

# Training using Gradient Descent
# train_step = tf.train.GradientDescentOptimizer(0.0000001)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(cost, {x:[1,2,3,4], y_:[0,-1,-2,-3]}))
