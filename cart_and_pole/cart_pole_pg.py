import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

gamma = .99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_aadd = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        # Feed forward part
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size,
                                      biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size,
                                          activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # Taining procedure
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outpuuts = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outpuuts)*self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(self.loss, tvars)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))

tf.reset_default
