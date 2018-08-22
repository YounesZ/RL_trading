import random

import matplotlib.pyplot as plt
import tensorflow as tf
import keras as K

from os import path
from src.lib import *
from src.emulator import Market
from src.simulators import Simulator
from src.sampler import SinSampler
from src.visualizer import Visualizer
# from src.logging import *
from copy import deepcopy
from datetime import datetime


class Agent:
    def __init__(self, input_size=32, action_size=1, layer_units=[48, 32, 16, 8], batch_size=12,\
                 discount_factor=0.95, buffer_size=200, learning_rate=0.001,\
                 prediction_model=None, planning_horizon=10, exploration_rate=0.05):

        self.model          =   QModelMLP(input_size, action_size, layer_units, learning_rate)
        self.batch_size     =   batch_size
        self.discount_factor=   discount_factor
        self.memory         =   []
        self.memory_short   =   []
        self.buffer_size    =   buffer_size
        self.planning_horizon = planning_horizon

    def remember(self, state, action, reward, next_state, done, next_valid_actions):
        self.memory_short.append((state, action, reward, next_state, done, next_valid_actions))
        if done:
            # Subtract mean
            rew_avg = np.mean([x[2] for x in self.memory_short])
            rew_std = np.std([x[2] for x in self.memory_short])
            # Transfer to long-term memory
            for ix in self.memory_short:
                temp = []
                temp = deepcopy((ix[0], ix[1], (ix[2] - rew_avg) / rew_std, ix[3], ix[4], ix[5]))
                self.memory.append(temp)
            self.memory_short = []

    def replay(self, window_size=100):
        if len(self.memory) > self.buffer_size:
            sample_id = [np.random.randint(len(self.memory)) for x in range(self.batch_size)]
            rewards = [self.get_discounted_reward(x, window_size) for x in sample_id]
            batch = [self.memory[x][0].flatten() for x in sample_id]

            self.model.fit(batch, [], rewards)


    def get_discounted_reward(self, bin_id, window_size):
        max_bin = np.minimum(bin_id + self.batch_size, (int(bin_id / window_size) + 1) * window_size)
        batch = self.memory[bin_id:max_bin]
        rews = [x[2].astype(float) for x in batch]
        disc = np.power(self.discount_factor, range(len(batch)))
        return np.sum(np.multiply(disc, rews))

    def get_q_valid(self, state, valid_actions):
        #self.model.model.set_weights(self.model.model.get_weights())
        q = self.model.predict(state.T)

        # assert np.max(q)<=1 and np.min(q)>=-1
        """
        # Constrain value within range
        q_valid = [np.nan] * len(q)
        for action in valid_actions:
            q_valid[action] = q[action]
        """
        return q  # q_valid

    def act(self, state, exploration, valid_actions):
        if np.random.random() > exploration:
            q_valid = self.get_q_valid(state, valid_actions)[0]
            return np.maximum(np.minimum(q_valid, valid_actions[1]), valid_actions[0])
        return np.random.random(1) + valid_actions[0]

    def save(self, fld):
        makedirs(fld)

        attr = {
            'batch_size': self.batch_size,
            'discount_factor': self.discount_factor,
            # 'memory':self.memory
        }

        pickle.dump(attr, open(os.path.join(fld, 'agent_attr.pickle'), 'wb'))
        self.model.save(fld)

    def load(self, fld):
        path = os.path.join(fld, 'agent_attr.pickle')
        print(path)
        attr = pickle.load(open(path, 'rb'))
        for k in attr:
            setattr(self, k, attr[k])
        self.model.load(fld)


def add_dim(x, shape):
    return np.reshape(x, (1,) + shape)



class QModelMLP():
    # multi-layer perception (MLP), i.e., dense only

    def __init__(self, state_shape, n_action, n_units, learning_rate):
        self.state_shape    =   state_shape
        self.n_action       =   n_action
        self.attr2save      =   ['state_shape', 'n_action', 'model_name']
        self.qmodel         =   'MLP'
        self.model, self.model_name  =   self.build_model(n_units, learning_rate)

    def build_model(self, n_units, learning_rate, activation='relu'):
        # if self.wavelet_channels==0:
        # Purely dense MLP
        model = K.models.Sequential()
        model.add(K.layers.Dense(n_units[0], input_shape=(np.prod(self.state_shape),)))

        for i in range(1, len(n_units)):
            model.add(keras.layers.Dense(n_units[i], activation=activation))
        # model.add(keras.layers.Dropout(drop_rate))
        model.add(keras.layers.Dense(self.n_action, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
        return model, self.qmodel + str(n_units)

    def save(self, fld):
        makedirs(fld)
        with open(os.path.join(fld, 'model.json'), 'w') as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights(os.path.join(fld, 'weights.hdf5'))

        attr = dict()
        for a in self.attr2save:
            attr[a] = getattr(self, a)
        pickle.dump(attr, open(os.path.join(fld, 'Qmodel_attr.pickle'), 'wb'))

    def load(self, fld, learning_rate):
        json_str = open(os.path.join(fld, 'model.json')).read()
        self.model = keras.models.model_from_json(json_str)
        self.model.load_weights(os.path.join(fld, 'weights.hdf5'))
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

        attr = pickle.load(open(os.path.join(fld, 'Qmodel_attr.pickle'), 'rb'))
        for a in attr:
            setattr(self, a, attr[a])

    def predict(self, state):
        q = self.model.predict(state)[0]
        if np.isnan(np.max(q, axis=1)).any():
            print('state'+str(state))
            print('q'+str(q))
            raise ValueError
        return q

    def fit(self, state, action, q_action):
        q = self.predict(state.T)
        q[action] = q_action
        self.model.fit(np.array(state), q, epochs=1, verbose=0)

def load_model(fld, learning_rate):
    qmodel  =   QModelMLP(None, None)
    qmodel.load(fld, learning_rate)
    return qmodel


def process_rewards(rew, reward, win_size, gamma):
    # Append to raw reward
    rew[0] += [reward]
    # Slice the data
    inf_lim = np.minimum(len(rew[0]), win_size)
    slice = rew[0][-inf_lim:]
    # Compute the moving average
    rew[1] += [np.mean(slice)]
    # Compute the exponential average
    e_avg = [x * np.exp(-ix * gamma) for ix, x in enumerate(slice[::-1])] / np.sum(
        [np.exp(-ix * gamma) for ix in range(len(slice))])
    rew[2] += [e_avg[0][0]]
    return rew


def viz_perf(x, rewards, ax):
    # Make a figure
    colors = ['r', 'b', 'k']
    labels = ['raw', 'moving average', 'exp. average']
    [ax.plot(range(x), ix, ic, label=il) for ix, ic, il in zip(rewards, colors, labels)]
    return


def test_sine():
    """
    	it is recommended to generate database using sampler.py before run main
    	"""

    # --- Agent's options
    batch_size          =   8
    learning_rate       =   1e-4
    discount_factor     =   0.8
    exploration_decay   =   0.99
    exploration_rate    =   0.01
    buffer_size         =   2000
    layer_units         =   [48, 32, 16, 8]
    fld_load            =   None
    exploration_init    =   1
    exploration_min     =   0.01

    # --- Environment's options
    rootStore           =   open('../dbloc.txt', 'r').readline().rstrip('\n')
    window_state        =   32
    time_difference     =   True
    wavelet_channels    =   0
    n_episode_training  =   100
    n_episode_testing   =   50
    open_cost           =   3  # Percentage on the buy order
    db_type             =   'SinSamplerDB';
    db                  =   'concat_half_base_';
    Sampler             =   SinSampler
    # db_type = 'PairSamplerDB'; db = 'randjump_100,1(10, 30)[]_'; Sampler = PairSampler

    fld     =   os.path.join(rootStore, 'data', db_type, db + 'A')
    sampler =   Sampler('single', 180, .5, (30, 35), (49, 50), fld=fld)
    env     =   Market(sampler, window_state, open_cost, time_difference=time_difference, wavelet_channels=wavelet_channels)
    agent   =   Agent(layer_units=layer_units, batch_size=batch_size,\
                 discount_factor=discount_factor, buffer_size=buffer_size, learning_rate=learning_rate,\
                 prediction_model=None, exploration_rate=exploration_rate)

    # Set save name
    fld_save    =   os.path.join(rootStore, 'results', sampler.title, 'DQN',\
                            str((env.window_state, sampler.window_episode, batch_size, learning_rate,\
                                 discount_factor, exploration_decay, env.open_cost)))

    visualizer = Visualizer(env.action_labels)

    print('=' * 20)
    print(fld_save)
    print('=' * 20)

    simulator =     Simulator(agent, env, visualizer=visualizer, fld_save=fld_save)
    simulator.train(n_episode_training, save_per_episode=1, exploration_decay=exploration_decay,
                    exploration_min=exploration_min, print_t='t', exploration_init=exploration_init)
    # agent.model = load_model(os.path.join(fld_save,'model'), learning_rate)

    # print('='*20+'\nin-sample testing\n'+'='*20)
    simulator.test(n_episode_testing, save_per_episode=1, subfld='in-sample testing')



# ========
# LAUNCHER
# ========
if __name__ == '__main__':
    test_sine()