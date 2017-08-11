import gym
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

min_reward    = -200
max_reward    = 200

### MODEL INIT ###
# print(env.observation_space)
# print(env.action_space)
STATE_SIZE  = 8
ACTION_SIZE = 4
state = tf.placeholder(shape=[None, STATE_SIZE], dtype=tf.float32)
hidden1 = tf.slim.fully_connected(state, H1, activation_fn=tf.nn.softmax)
hidden2 = tf.slim.fully_connected(H1, H2, activation_fn=tf.nn.softmax)
output = tf.slim.fully_connected(H2, ACTION_SIZE, activation_fn=tf.nn.softmax)

chosen_action = tf.argmax(self.output, 1)

### Training Procedure ###
reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

observation = env.reset()
for _ in range(1000):
    env.render()
    # print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
        break
