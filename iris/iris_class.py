import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import random

### Number of features
n = 4
classes = 3

### Import data
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

### Set up TF variables
x = tf.placeholder(tf.float32, [None, n])
y = tf.placeholder(tf.float32, [None, classes])
W = tf.Variable(tf.zeros([n, classes]))
b = tf.Variable(tf.zeros([classes]))

h = tf.nn.softmax(tf.matmul(x, W) + b)

### Cost function 1/(m) * sum((y_-y)**2) ###
cost = -tf.reduce_mean(tf.reduce_sum(y*tf.log(h)+(1-y)*tf.log(1-h), reduction_indices=[1]))
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), reduction_indices=[1]))


### Training using Gradient Descent ###
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

### Training Loop ###
steps = 1000
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(steps):
    sess.run(optimizer, {x:x_ds, y:y_ds})
    print("After %d iterations:" % int(i+1))
    print("W: %s" % sess.run(W))
    print("b: %s" % sess.run(b))

### Results ###
# curr_W = sess.run(W)
# curr_b = sess.run(b)
# curr_W, curr_b, curr_loss = sess.run([W, b, cost], feed_dict={x:x_ds, y:y_ds})
# print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(h,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Classified with %f accuracy" % sess.run(accuracy, feed_dict={x: x_ts, y: y_ts}))
