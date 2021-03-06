import random

import matplotlib.pyplot as plt
import tensorflow as tf
import keras as K

from os import path
from src.lib import *
#from src.logging import *
from copy import deepcopy
from datetime import datetime


class Agent:

	def __init__(self, model, batch_size=12, discount_factor=0.95,
				 buffer_size=200, prediction_model=None, planning_horizon=10):

		self.model 		= 	model
		self.p_model 	=	prediction_model
		self.batch_size = 	batch_size
		self.discount_factor = discount_factor
		self.memory 	= 	[]
		self.memory_short=	[]
		self.buffer_size=	buffer_size
		self.planning_horizon 	=	planning_horizon


	def remember(self, state, action, reward, next_state, done, next_valid_actions):
		self.memory_short.append( (state, action, reward, next_state, done, next_valid_actions) )
		if done:
			# Subtract mean
			rew_avg		=	np.mean( [x[2] for x in self.memory_short] )
			rew_std 	=	np.std( [x[2] for x in self.memory_short] )
			# Transfer to long-term memory
			for ix in self.memory_short:
				temp 	= 	[]
				temp 	=	deepcopy( (ix[0], ix[1], (ix[2]-rew_avg)/rew_std, ix[3], ix[4], ix[5]) )
				self.memory.append( temp )
			self.memory_short=	[]


	def replay(self, window_size=100):
		if len(self.memory)>self.buffer_size:
			sample_id 	=	[np.random.randint(len(self.memory)) for x in range(self.batch_size)]
			rewards		=	[self.get_discounted_reward(x, window_size) for x in sample_id]
			batch 		= 	[self.memory[x][0].flatten() for x in sample_id]

			self.model.fit(batch, [], rewards)

			"""
			for state, action, reward, next_state, done, next_valid_actions in batch:
				self.model.fit(np.concatenate([x[0] for x in batch], axis=1), [], [x[2] for x in batch])
				q = deepcopy(reward)
				if not done:
					q += self.discount_factor * np.nanmax(self.get_q_valid(next_state, next_valid_actions))
				self.model.fit(state, action, q)
			"""


	def get_discounted_reward(self, bin_id, window_size):
		max_bin =	np.minimum( bin_id+self.batch_size, ( int( bin_id/window_size ) + 1 ) * window_size )
		batch 	=	self.memory[bin_id:max_bin]
		rews 	=	[x[2].astype(float) for x in batch]
		disc 	=	np.power(self.discount_factor, range(len(batch)))
		return np.sum( np.multiply(disc, rews) )


	def get_q_valid(self, state, valid_actions):
		self.p_model.model.set_weights(self.model.model.get_weights())
		q 	=	self.p_model.predict(state.T)

		#assert np.max(q)<=1 and np.min(q)>=-1
		"""
		# Constrain value within range
		q_valid = [np.nan] * len(q)
		for action in valid_actions:
			q_valid[action] = q[action]
		"""
		return q 		#q_valid


	def act(self, state, exploration, valid_actions):
		if np.random.random() > exploration:
			q_valid = self.get_q_valid(state, valid_actions)[0]
			return np.maximum( np.minimum(q_valid, valid_actions[1]), valid_actions[0] )
		return np.random.random(1) + valid_actions[0]


	def save(self, fld):
		makedirs(fld)

		attr = {
			'batch_size':self.batch_size, 
			'discount_factor':self.discount_factor, 
			#'memory':self.memory
			}

		pickle.dump(attr, open(os.path.join(fld, 'agent_attr.pickle'),'wb'))
		self.model.save(fld)


	def load(self, fld):
		path = os.path.join(fld, 'agent_attr.pickle')
		print(path)
		attr = pickle.load(open(path,'rb'))
		for k in attr:
			setattr(self, k, attr[k])
		self.model.load(fld)



def add_dim(x, shape):
	return np.reshape(x, (1,) + shape)



class QModelKeras:
	# ref: https://keon.io/deep-q-learning/
	
	def init(self):
		pass

	def build_model(self):
		pass

	def __init__(self, state_shape, n_action, wavelet_channels=0):
		self.state_shape	=	state_shape
		self.n_action 		= 	n_action
		self.attr2save 		= 	['state_shape','n_action','model_name']
		self.wavelet_channels=	wavelet_channels
		self.init()

	def save(self, fld):
		makedirs(fld)
		with open(os.path.join(fld, 'model.json'), 'w') as json_file:
			json_file.write(self.model.to_json())
		self.model.save_weights(os.path.join(fld, 'weights.hdf5'))

		attr = dict()
		for a in self.attr2save:
			attr[a] = getattr(self, a)
		pickle.dump(attr, open(os.path.join(fld, 'Qmodel_attr.pickle'),'wb'))

	def load(self, fld, learning_rate):
		json_str = open(os.path.join(fld, 'model.json')).read()
		self.model = keras.models.model_from_json(json_str)
		self.model.load_weights(os.path.join(fld, 'weights.hdf5'))
		self.model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))

		attr = pickle.load(open(os.path.join(fld, 'Qmodel_attr.pickle'), 'rb'))
		for a in attr:
			setattr(self, a, attr[a])

	def predict(self, state):
		# UNCOMMENT for modular wavelet channels
		"""
		# Reshape state-space if wavelet transformed
		if self.wavelet_channels>0:
			rshp_state 	= 	self.modular_state_space(state)
		else:
			rshp_state	=	add_dim(state, self.state_shape)
		"""
		#rshp_state 	= 	add_dim(state, self.state_shape)
		q 			= 	self.model.predict( state )[0]

		"""
		if np.isnan(np.max(q, axis=1)).any():
			print('state'+str(state))
			print('q'+str(q))
			raise ValueError
		"""

		return q

	def fit(self, state, action, q_action):
		"""
		q = self.predict(state.T)
		q[action] = q_action
		"""
		q = q_action
		# UNCOMMENT for modular wavelet channels
		"""
		if self.wavelet_channels>0:
			rshp_state	=	self.modular_state_space(state)
		else:
			rshp_state	=	add_dim(state, self.state_shape)
		self.model.fit( rshp_state, add_dim(q, (self.n_action,)), epochs=1, verbose=0)
		"""
		self.model.fit( np.array(state), q, epochs=1, verbose=0)

	def modular_state_space(self, state):
		new_shape 	= 	(len(state) / np.power(2, list(range(1, self.wavelet_channels + 1)) + [self.wavelet_channels])).astype(int)
		cum_shape 	= 	np.cumsum(new_shape).astype(int)
		# rshp_state 		=	[add_dim(state[x:y], (z,1)) for x,y,z in zip( np.insert(cum_shape,0,0), cum_shape, new_shape )]
		rshp_state 	= 	[state[x:y].T for x, y in zip(np.insert(cum_shape[:-1], 0, 0), cum_shape)]
		return rshp_state


class QModelMLP(QModelKeras):
	# multi-layer perception (MLP), i.e., dense only

	def init(self):
		self.qmodel = 'MLP'	

	def build_model(self, n_units, learning_rate, activation='relu'):

		#if self.wavelet_channels==0:
		# Purely dense MLP
		model = K.models.Sequential()
		model.add( K.layers.Dense( n_units[0], input_shape=(np.prod(self.state_shape),)) )

		for i in range(1, len(n_units)):
			model.add(keras.layers.Dense(n_units[i], activation=activation))
			#model.add(keras.layers.Dropout(drop_rate))

		model.add(keras.layers.Dense(self.n_action, activation='linear'))

		"""
		else:
			# Composite architecture : 1MLP for each decomposition channel
			max_scales 	=	int(np.log(self.state_shape[0])/np.log(2))
			inputs 		=	np.power(2, range(1, self.wavelet_channels+1))[max_scales-self.wavelet_channels-1:][::-1]
			inp_layers 	=	[]
			inp_models 	=	[]
			for ii in np.append(inputs, inputs[-1]):
				# Make a model
				input 	=	keras.layers.Input(shape=(ii,))
				hid1 	= 	keras.layers.Dense(int(1.5*ii), activation='relu')(input)
				hid2 	=	keras.layers.Dense(int(1.5*ii), activation='relu')(hid1)
				inp_layers 	+=	[input]
				inp_models 	+=	[hid2]
			# Make composite
			composite 	= 	keras.layers.Concatenate(axis=-1)(inp_models)
			compoD1 	= 	keras.layers.Dense(self.state_shape[0])(composite)
			compoD2 	= 	keras.layers.Dense( int(self.state_shape[0]*.75) )(compoD1)
			output 		=	keras.layers.Dense(self.n_action, activation='linear')(compoD2)
			model 		=	keras.models.Model(inputs=inp_layers, outputs=output)
		"""

		model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
		self.model 		= 	model
		self.model_name = 	self.qmodel + str(n_units)


class PGModelMLP(QModelKeras):
	# multi-layer perception (MLP), i.e., dense only

	def init(self):
		self.qmodel = 'MLP_PG'

	def build_model(self, n_hidden, learning_rate, activation='relu', input_size=32, batch_size=8):

		# --- Purely dense MLP
		input 	=	keras.layers.Input((batch_size, input_size), name='main_input')
		# Hidden
		hidden 	=	[]
		in_lay 	=	input
		for ii in n_hidden:
			hidden 	+=	[keras.layers.Dense(units=ii, activation=activation)(in_lay)]
			in_lay 	=	hidden[-1]
		# Output
		#output1	=	keras.layers.Dense(units=self.n_action, activation='linear')(in_lay)
		#output2 =	keras.layers.Dense(units=self.n_action, activation='softmax')(output1)
		output1 	= 	keras.layers.Dense(units=self.n_action, activation='tanh')(in_lay)
		model 		= 	keras.models.Model(inputs=input, outputs=output1)
		#model 	= 	keras.models.Model(inputs=input, outputs=(output1, output2) )

		#model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=learning_rate))
		model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
		self.model 			= 	model
		self.model_name 	= 	self.qmodel + str(n_hidden)
		self.learning_rate 	=	learning_rate



class QModelRNN(QModelKeras):
	"""
	https://keras.io/getting-started/sequential-model-guide/#example
	note param doesn't grow with len of sequence
	"""

	def _build_model(self, Layer, n_hidden, dense_units, learning_rate, activation='relu'):

		model = keras.models.Sequential()
		model.add(keras.layers.Reshape(self.state_shape, input_shape=self.state_shape))
		m = len(n_hidden)
		for i in range(m):
			model.add(Layer(n_hidden[i],
				return_sequences=(i<m-1)))
		for i in range(len(dense_units)):
			model.add(keras.layers.Dense(dense_units[i], activation=activation))
		model.add(keras.layers.Dense(self.n_action, activation='linear'))
		model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
		self.model = model
		self.model_name = self.qmodel + str(n_hidden) + str(dense_units)


class QModelLSTM(QModelRNN):
	def init(self):
		self.qmodel = 'LSTM'
	def build_model(self, n_hidden, dense_units, learning_rate, activation='relu'):
		Layer = keras.layers.LSTM
		self._build_model(Layer, n_hidden, dense_units, learning_rate, activation)


class QModelGRU(QModelRNN):
	def init(self):
		self.qmodel = 'GRU'
	def build_model(self, n_hidden, dense_units, learning_rate, activation='relu'):
		Layer = keras.layers.GRU
		self._build_model(Layer, n_hidden, dense_units, learning_rate, activation)


class QModelConv(QModelKeras):
	"""
	ref: https://keras.io/layers/convolutional/
	"""
	def init(self):
		self.qmodel = 'Conv'

	def build_model(self, 
		filter_num, filter_size, dense_units, 
		learning_rate, activation='relu', dilation=None, use_pool=None):

		if use_pool is None:
			use_pool = [True]*len(filter_num)
		if dilation is None:
			dilation = [1]*len(filter_num)

		model = keras.models.Sequential()
		model.add(keras.layers.Reshape(self.state_shape, input_shape=self.state_shape))
		
		for i in range(len(filter_num)):
			model.add(keras.layers.Conv1D(filter_num[i], kernel_size=filter_size[i], dilation_rate=dilation[i], 
				activation=activation, use_bias=True))
			if use_pool[i]:
				model.add(keras.layers.MaxPooling1D(pool_size=2))
		
		model.add(keras.layers.Flatten())
		for i in range(len(dense_units)):
			model.add(keras.layers.Dense(dense_units[i], activation=activation))
		model.add(keras.layers.Dense(self.n_action, activation='linear'))
		model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
		
		self.model = model

		self.model_name = self.qmodel + str([a for a in
			zip(filter_num, filter_size, dilation, use_pool)
			])+' + '+str(dense_units)
		

class QModelConvRNN(QModelKeras):
	"""
	https://keras.io/getting-started/sequential-model-guide/#example
	note param doesn't grow with len of sequence
	"""

	def _build_model(self, RNNLayer, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate, 
		conv_kernel_size=3, use_pool=False, activation='relu'):

		model = keras.models.Sequential()
		model.add(keras.layers.Reshape(self.state_shape, input_shape=self.state_shape))

		for i in range(len(conv_n_hidden)):
			model.add(keras.layers.Conv1D(conv_n_hidden[i], kernel_size=conv_kernel_size, 
				activation=activation, use_bias=True))
			if use_pool:
				model.add(keras.layers.MaxPooling1D(pool_size=2))
		m = len(RNN_n_hidden)
		for i in range(m):
			model.add(RNNLayer(RNN_n_hidden[i],
				return_sequences=(i<m-1)))
		for i in range(len(dense_units)):
			model.add(keras.layers.Dense(dense_units[i], activation=activation))

		model.add(keras.layers.Dense(self.n_action, activation='linear'))
		model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=learning_rate))
		self.model = model
		self.model_name = self.qmodel + str(conv_n_hidden) + str(RNN_n_hidden) + str(dense_units)
		

class QModelConvLSTM(QModelConvRNN):
	def init(self):
		self.qmodel = 'ConvLSTM'
	def build_model(self, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate, 
		conv_kernel_size=3, use_pool=False, activation='relu'):
		Layer = keras.layers.LSTM
		self._build_model(Layer, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate, 
		conv_kernel_size, use_pool, activation)


class QModelConvGRU(QModelConvRNN):
	def init(self):
		self.qmodel = 'ConvGRU'
	def build_model(self, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate, 
		conv_kernel_size=3, use_pool=False, activation='relu'):
		Layer = keras.layers.GRU
		self._build_model(Layer, conv_n_hidden, RNN_n_hidden, dense_units, learning_rate, 
		conv_kernel_size, use_pool, activation)


def load_model(fld, learning_rate):
	s = open(os.path.join(fld,'QModel.txt'),'r').read().strip()
	qmodels = {
		'Conv':QModelConv,
		'DenseOnly':QModelMLP,
		'MLP':QModelMLP,
		'LSTM':QModelLSTM,
		'GRU':QModelGRU,
		}
	qmodel = qmodels[s](None, None)
	qmodel.load(fld, learning_rate)
	return qmodel





def test_ddpg():
	import gym

	sess	=	tf.Session()
	K.backend.set_session(sess)
	env 	= 	gym.make("Pendulum-v0")
	actor_critic=	DDPGModelMLP(sess, exploration_rate=0.05, buffer_size=200, tau=0.8)
	actor_critic.build_graph(sess, [16, 16, 12, 6], [[16, 16], [16, 10]], 1e-4, env.observation_space.shape, env.action_space.shape)
	num_trials	=	10000
	trial_len  	=	500

	cur_state	=	env.reset()
	n_steps 	=	1
	all_rew		=	[[], [], []]
	FF 			= 	plt.figure()
	AX 			= 	FF.add_subplot(111)
	plt.ion()
	plt.draw()

	while True:
		print('Step {}'.format(n_steps))#env.render()
		cur_state	=	cur_state.reshape((1, env.observation_space.shape[0]))
		action		=	actor_critic.act(cur_state, env.action_space.sample())
		action		=	action.reshape( (1, env.action_space.shape[0]) )

		new_state, reward, done, _	=	env.step(action)
		new_state 	= 	new_state.reshape( (1, env.observation_space.shape[0]) )

		actor_critic.remember(cur_state, action, reward, new_state, done)
		actor_critic.replay()

		cur_state 	= 	new_state

		# Plot rewards
		all_rew 	=	process_rewards(all_rew, reward, 20, 0.5)
		if n_steps%20==0:
			viz_perf( n_steps, all_rew, AX)
			plt.pause(0.05)
			FF.canvas.draw()
		n_steps += 1


def process_rewards(rew, reward, win_size, gamma):
	# Append to raw reward
	rew[0] 	+=	[reward]
	# Slice the data
	inf_lim	=	np.minimum(len(rew[0]), win_size)
	slice 	=	rew[0][-inf_lim:]
	# Compute the moving average
	rew[1]	+=	[np.mean(slice)]
	# Compute the exponential average
	e_avg 	=	[x*np.exp(-ix*gamma) for ix, x in enumerate(slice[::-1])] / np.sum([np.exp(-ix*gamma) for ix in range(len(slice))])
	rew[2] 	+=	[e_avg[0][0]]
	return rew

def viz_perf(x, rewards, ax):
	# Make a figure
	colors 		=	['r', 'b', 'k']
	labels 		= 	['raw', 'moving average', 'exp. average']
	[ax.plot(range(x), ix, ic, label=il) for ix, ic, il in zip(rewards, colors, labels)]
	return


# ========
# LAUNCHER
# ========
if __name__ == '__main__':
	test_ddpg()