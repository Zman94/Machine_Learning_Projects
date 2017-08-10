import numpy as np
import _pickle as pickle
import gym

# hyperparameters
H = 200 # hidden layer neurons
batch_size = 10 # how many episodes before update
learning_rate = 1e-4
gamma = .99 # discount factor
decay_rate = .99 # decay factor
resume = True # resume from previous checkpoint
render = False

# model initialization
D = 80*80 # input dimensionality
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" intialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k : np.zeros_like(v) for k, v in model.items()} # update buffers that add up gradients over a batch
rmsprop_cache = {k : np.zeros_like(v) for k, v in model.items()} # rmsprop memory

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1d float vector """
    I = I[35:195]
    I = I[::2,::2,0]
    I[I == 144] = 0 # erase background
    I[I == 109] = 0
    I[I != 0] = 1 # everything else is 1
    return I.astype(np.float).ravel()

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 00:
            running_add = 0 # reset the sum since boundary
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 # ReLu
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h # return probability for taking action 2

def policy_backward(eph, epdlogp):
    """ backward pass. (eph -- array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0 # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1' : dW1, 'W2' : dW2}

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # computing difference frame
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render:
        env.render()

    # preprocess the observation
    # set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3

    # record various intermediates (for backprop)
    xs.append(x)
    hs.append(h)
    y = 1 if action == 2 else 0 # a "fake label"
    dlogps.append(y - aprob) # grad that encourages the action

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward) # record reward

    if done: # episode finish
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize rewards to be unit normal
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k]

        # parameter update
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)

        # book keeping
        running_reward = reward_sum if running_reward is None else running_reward*.99 + reward_sum*.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0:
            pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()
        prev_x = None

    if reward != 0:
        print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!'))
