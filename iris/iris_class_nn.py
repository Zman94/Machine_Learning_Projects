import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import random

n = 4
classes = 3

### Import Data ###
type_to_number = {'Iris-setosa':[1,0,0], 'Iris-versicolor':[0,1,0], 'Iris-virginica':[0,0,1]}
iris_data = []
with open("iris_data.txt") as f:
    for line in f:
        iris_data.append(line.strip())

x_ds = []
y_ds = []
x_ts = []
y_ts = []
random.shuffle(iris_data)
test_data = iris_data[-40:]
iris_data = iris_data[:-40]
for i in iris_data:
    line_data = i.split(',')
    x_data = [float(j) for j in line_data[:-1]]
    x_ds.append(x_data)
    y_ds.append(line_data[-1])
for i in test_data:
    line_data = i.split(',')
    x_data = [float(j) for j in line_data[:-1]]
    x_ts.append(x_data)
    y_ts.append(line_data[-1])
for i in range(len(y_ds)):
    y_ds[i] = type_to_number[y_ds[i]]
for i in range(len(y_ts)):
    y_ts[i] = type_to_number[y_ts[i]]

### Set up TF variables ###
images = tf.placeholder(tf.float32, shape=(batch_size, n))
# labels = tf.placeholder(tf.float32, shape=(batch_size))

### Build Model for Inference ###
with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([n, hidden1_units],
                            stddev=1.0 / math.sqrt(float(n))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                        name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights)+biases)

with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                        name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights)+biases)

with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, classes],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([classes]),
                        name='biases')
    logits = tf.matmul(hidden2, weights)+biases

### Calculate loss ###
labels = tf.to_int64(labels)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labels, logits=logits, name='xentropy')
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

### Training ###
tf.summary.scalar('loss', loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
global_step = tf.variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)

### Evaluate ###


# W = tf.Variable(tf.zeros([n, classes]))
# b = tf.Variable(tf.zeros([classes]))

