import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np
from gym import wrappers
env = gym.make('LunarLander-v2')
# env = wrappers.Monitor(env, './lunar_lander-experiment-1', force=True)

### HYPER PARAMETERS ###
H1            = 50
H2            = 200
batch_size    = 10
learning_rate = 1e-3
gamma         = .99
decay_rate    = .99
resume        = True
render        = True
score_req     = 50

min_reward    = -200
max_reward    = 200

STATE_SIZE  = 8
ACTION_SIZE = 4

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []






### MODEL INIT ###
# print(env.observation_space)
# print(env.action_space)
state = tf.placeholder(shape=[None, STATE_SIZE], dtype=tf.float32)
output = tf.placeholder(shape=[None, ACTION_SIZE], dtype=tf.float32)

batch_size = 10
learning_rate = 1e-4

weights = {
    'hidden1' : tf.Variable(tf.random_normal([STATE_SIZE, H1])),
    # 'hidden2' : tf.Variable(tf.random_normal([H1, H2])),
    'output'  : tf.Variable(tf.random_normal([H1, ACTION_SIZE]))
}

biases = {
    'hidden1' : tf.Variable(tf.random_normal([H1])),
    # 'hidden2' : tf.Variable(tf.random_normal([H2])),
    'output'  : tf.Variable(tf.random_normal([ACTION_SIZE]))
}

hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(state, weights['hidden1']), biases['hidden1']))
# hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer1, weights['hidden2']), biases['hidden2']))
output_layer  = tf.matmul(hidden_layer1, weights['output']) + biases['output']

cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=output))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

chosen_action = tf.argmax(output, 1)

### Training Procedure ###
reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    observation = env.reset()
    running_reward = 0
    ep_history = []
    gradBuffer = sess.run(tf.trainable_variables())
    while True:
        if render:
            env.render()
        print(observation.shape)
        print(state)

        action = sess.run(chosen_action, feed_dict={state:[observation]})

        s1, r, done, _ = env.step(a_dist)
        ep_history.append(observation, action, r, s1)
        running_reward += r

        if done:
            observation = env.reset()
            break
