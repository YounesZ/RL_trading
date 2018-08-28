import random
import keras as K
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from os import path
from copy import deepcopy
from datetime import datetime
from itertools import compress
from keras.engine.input_layer import Input
from keras.backend import variable as Kvar
from keras.backend import set_value as Kset
from generic.dummy import Noise as dumnz
from generic.noise import OrnsteinUhlenbeckActionNoise


class Agent:

	def __init__(self, agent_type, state_size, action_size, valid_actions=None,
				 layer_units=[48, 32, 16, 8], batch_size=12, discount_factor=0.95, buffer_min_size=200,
				 buffer_max_size=2000, learning_rate=0.001, noise_process=None, outputdir=None):

		# Set action space
		self.valid_actions	=	list( range(action_size) )
		self.batch_size     =   batch_size
		self.discount_factor=   discount_factor
		self.memory         =   []
		self.buffer_min_size=   buffer_min_size
		self.buffer_max_size=   buffer_max_size
		self.attr2save      =   ['state_size', 'action_size', 'model_name']
		self.outputdir 		=	outputdir
		self.model          =   self.get_agent(agent_type, state_size, action_size, layer_units, learning_rate, discount_factor, noise_process)


	def get_agent(self, agent_type, state_size, action_size, layer_units, learning_rate, discount_factor, noise_process):
		if agent_type	==	'DQN':
			return	DQN_agent(state_size, action_size, layer_units, learning_rate, discount_factor, self.outputdir)
		elif agent_type	==	'DDPG':
			return DDPG_agent(state_size, action_size, actor_hidden=layer_units, critic_hidden=layer_units, noise_process=noise_process, outputdir=self.outputdir, discount_factor=discount_factor)
		else:
			print('Unrecognized agent type')
			return


	def remember(self, state, action, reward, next_state, done, next_valid_actions=None):
		if next_valid_actions is None:
			next_valid_actions 	=	self.valid_actions
		self.memory.append((state, action, reward, next_state, done, next_valid_actions))


	def replay(self, log=False, learning_rate=1e-3):
		Q_s_a 	=	[]
		Qloss 	=	[]
		if len(self.memory) > self.buffer_min_size:
			batch   	=   random.sample(self.memory, min(len(self.memory), self.batch_size))
			state 		=	np.concatenate([x[0] for x in batch])
			action 		= 	np.reshape([x[1] for x in batch], [-1,1])
			reward 		= 	np.reshape([x[2] for x in batch], [-1,1])
			next_state	= 	np.concatenate([x[3] for x in batch])
			done 		= 	np.reshape([x[4] for x in batch], [-1,1])
			next_valid	=	np.concatenate([np.reshape(x[5], [1,-1]) for x in batch])
			Qloss		=	self.model.fit(state, action, reward, next_state, done, next_valid=next_valid, lr=learning_rate, log=log)
		return Q_s_a, Qloss


	def act(self, state, exploration=0.05, valid_actions=None):
		# NEED TO PLUG NOISE PROCESS HERE
		if valid_actions is None:
			valid_actions	=	self.valid_actions
		if np.random.random() > exploration:
			action 	=	self.model.act(state, valid_actions=valid_actions) #+ self.model.noise_process()
		elif self.model.action_size>1:
			action 	=	np.random.choice( valid_actions )
		else:
			action 	=	np.reshape(np.random.random()*2-1, [1,1])
		return action


	def predict(self, state):
		return self.model.predict(state)


	def save(self, folder):
		pass	# for now. ..


class DDPG_agent():
	# Actor-critic method, both are implemented as MLPs

	def __init__(self, state_space, action_space, \
				 exploration_rate	=	0.05,\
				 batch_size			=	64,\
				 tau				=	0.001,\
				 discount_factor	=	0.99,\
				 noise_process 		=	None,\
				 outputdir			=	None,\
				 actor_hidden		=	[400, 300],\
				 critic_hidden		=	[400, 300]):
		self.qmodel             =   'DDPG'
		self.session            =   tf.Session()
		self.state_size        	=   state_space
		self.action_size      	=   action_space
		self.exploration_rate 	=	exploration_rate
		self.batch_size 		=	batch_size
		self.update_rate 		=	tau
		self.discount_rate 		=	discount_factor
		self.random_seed 		=	np.random.randint(1e9, size=6)
		self.memory 			=	[]
		self.model_saver 		=	None
		self.outputdir 			=	outputdir
		# Set noise process
		if noise_process=='OrnsteinUhlenbeck':
			self.noise_process 	=	OrnsteinUhlenbeckActionNoise( mu=np.zeros(action_space) )
		elif noise_process=='dummy':
			self.noise_process 	=	dumnz
		# Make the graph
		self.actor_hidden       =   actor_hidden
		self.critic_hidden      =   critic_hidden
		self.build_graph()


	def build_graph(self):
		weight_count 	=	0

		# ===============
		# --- Setup actor
		# ===============

		# Build networks
		self.actor_input, self.actor_output, self.actor_training	=	self.build_actor(self.state_size, self.actor_hidden, self.action_size)
		self.actor_weights 			= 	tf.trainable_variables()
		weight_count 				+=	len(self.actor_weights)
		self.actor_target_input, self.actor_target_output, self.actor_target_training	=	self.build_actor(self.state_size, self.actor_hidden, self.action_size, 'target')
		self.actor_target_weights 	= 	tf.trainable_variables()[weight_count:]
		weight_count 				+= 	len(self.actor_target_weights)

		# Set actor's learning rate
		with tf.name_scope('actor_lr'):
			self.lr_actor 			= 	tf.placeholder(tf.float32)
			tf.summary.scalar('actor_learning_rate', self.lr_actor)

		# Set op: weights update
		self.update_actor_target_params	= 	[x.assign( tf.multiply(y, self.update_rate) + tf.multiply(x, 1-self.update_rate) ) for x,y in zip(self.actor_target_weights, self.actor_weights)]

		# Set op: optimization
		self.actor_critic_grad 		=	tf.placeholder(tf.float32, [None, self.action_size], name="dQ_dCritic")
		unnorm_actor_grad			=	tf.gradients(self.actor_output, self.actor_weights, -self.actor_critic_grad, name="dAction_dActorParams")
		norm_actor_grad 			=	list(map(lambda x: tf.div(x, self.batch_size), unnorm_actor_grad))
		grads 						=	zip(norm_actor_grad, self.actor_weights)
		self.optimize_actor 		= 	tf.train.AdamOptimizer(self.lr_actor).apply_gradients(grads)


		# ================
		# --- Setup critic
		# ================

		# Build networks
		self.critic_state_input, self.critic_action_input, self.critic_output	=	self.build_critic(self.state_size, self.critic_hidden, self.action_size)
		self.critic_weights 		=	tf.trainable_variables()[weight_count:]
		weight_count 				+= 	len(self.critic_weights)
		self.critic_target_state_input, self.critic_target_action_input, self.critic_target_output	= 	self.build_critic(self.state_size, self.critic_hidden, self.action_size, 'target')
		self.critic_target_weights 	= 	tf.trainable_variables()[weight_count:]
		weight_count 				+= 	len(self.critic_target_weights)

		# Set critic's learning rate
		with tf.name_scope('critic_lr'):
			self.lr_critic			=	tf.placeholder(tf.float32)
			tf.summary.scalar('critic_learning_rate', self.lr_critic)

		# Set op: weights update
		self.update_critic_target_params	= 	[x.assign(tf.multiply(y, self.update_rate) + tf.multiply(x, 1 - self.update_rate)) for x, y in zip(self.critic_target_weights, self.critic_weights)]

		# Set op: optimization
		self.critic_grad 	= 	tf.gradients(self.critic_output, self.critic_action_input, name="dQ_dAction")
		self.y_input 		=	tf.placeholder("float", shape=[None, self.action_size], name="Qvalue_target")
		self.critic_loss	=	tf.reduce_mean( tf.square(self.y_input - self.critic_output), name="critic_loss" )
		self.optimize_critic=	tf.train.AdamOptimizer(self.lr_critic).minimize(self.critic_loss)


		# =================
		# --- Setup summary
		# =================

		# Add reward to graph summary
		with tf.name_scope('episode_total_reward'):
			self.episode_reward	=	tf.placeholder(tf.float32)
			tf.summary.scalar("episode_reward", self.episode_reward)

		# Add action value to graph summary
		with tf.name_scope('episode_max_Qvalue'):
			self.episode_maxQ	=	tf.placeholder(tf.float32)
			tf.summary.scalar('episode_max_value', self.episode_maxQ)

		# Add Qloss to graph summary
		with tf.name_scope('episode_totLoss'):
			self.episode_Qlss 	= tf.placeholder(tf.float32)
			tf.summary.scalar('episode_total_loss', self.episode_Qlss)
		self.merged_sum 	= 	tf.summary.merge_all()

		# --- Initialize graph
		if not self.outputdir is None:
			self.sumWriter 	=	tf.summary.FileWriter(self.outputdir, self.session.graph)
		self.model_saver	=	tf.train.Saver()
		self.session.run(tf.initialize_all_variables())


	def build_actor(self, input_size, actor_hidden, output_size, handle_suffix=''):
		with tf.name_scope('actor_'+handle_suffix):
			# Input layer
			state_input	=	tf.placeholder("float", [None, input_size], name="actor_state_input")
			is_training	=	tf.placeholder("bool", name="actor_training_flag")

			# --- VARIABLES
			# Hidden layer 1
			W1 	=	tf.Variable( tf.random_uniform([input_size, actor_hidden[0]],-3e-3,3e-3), name= 'ac_w1'+handle_suffix )
			b1 	=	tf.Variable( tf.random_uniform([actor_hidden[0]],-3e-3,3e-3), name='ac_b1'+handle_suffix )
			# Hidden layer 2
			W2 	= 	tf.Variable( tf.random_uniform([actor_hidden[0], actor_hidden[1]], -3e-3, 3e-3), name= 'ac_w2'+handle_suffix )
			b2 	= 	tf.Variable( tf.random_uniform([actor_hidden[1]], -3e-3, 3e-3), name='ac_b2'+handle_suffix )
			# Hidden layer 3
			W3 	= 	tf.Variable( tf.random_uniform([actor_hidden[1], output_size], -3e-3,3e-3), name= 'ac_w3'+handle_suffix )
			b3 	= 	tf.Variable( tf.random_uniform([output_size], -3e-3, 3e-3), name='ac_b3'+handle_suffix )

			# --- GRAPH
			#output 	=	self.batch_norm_layer(state_input, is_training, activation=tf.identity, scope_bn='batch_norm0'+handle_suffix)
			output 	=	state_input
			output 	=	tf.add( tf.matmul(output, W1), b1 )
			output 	=	tf.nn.relu( output )
			#output 	= 	self.batch_norm_layer(output, is_training, activation=tf.identity, scope_bn='batch_norm1'+handle_suffix)
			output 	=	tf.add( tf.matmul(output, W2), b2 )
			output 	=	tf.nn.relu(output)
			#output 	= 	self.batch_norm_layer(output, is_training, activation=tf.identity, scope_bn='batch_norm2'+handle_suffix)
			output 	=	tf.add( tf.matmul(output, W3), b3 )
			output 	=	tf.nn.tanh( output )
		return state_input, output, is_training


	def build_critic(self, input_size, critic_hidden, output_size, handle_suffix=''):
		with tf.name_scope('critic_' + handle_suffix):
			# Input layer 1: the state space
			state_input = 	tf.placeholder("float", shape=[None, input_size], name="critic_state_input")
			# Input layer 2: the action
			action_input=	tf.placeholder("float", shape=[None, output_size], name="critic_action_input")

			# --- VARIABLES
			# Hidden layer 1
			W1 	=	tf.Variable( tf.random_uniform([input_size, critic_hidden[0]], -3e-3, 3e-3), name= 'cr_w1'+handle_suffix )
			b1 	=	tf.Variable( tf.random_uniform([critic_hidden[0]], -3e-3, 3e-3), name= 'cr_b1'+handle_suffix )
			# Hidden layer 2: take up concatenation on action_input
			W2s	=	tf.Variable( tf.random_uniform([critic_hidden[0], critic_hidden[1]], -3e-3, 3e-3), name= 'cr_w2s'+handle_suffix  )
			W2a = 	tf.Variable( tf.random_uniform([output_size, critic_hidden[1]], -3e-3, 3e-3), name='cr_w2a' + handle_suffix)
			b2 	= 	tf.Variable( tf.random_uniform([critic_hidden[1]], -3e-3, 3e-3), name='cr_b2' + handle_suffix)
			# Output layer
			W3 	=	tf.Variable( tf.random_uniform([critic_hidden[1], output_size], -3e-3, 3e-3), name= 'cr_w3'+handle_suffix  )
			b3 	=	tf.Variable( tf.random_uniform([output_size], -3e-3, 3e-3), name= 'cr_b3'+handle_suffix )

			# --- GRAPH
			output 	=	tf.add( tf.matmul(state_input, W1), b1 )
			output 	=	tf.nn.relu( output )
			t1 		=	tf.matmul(output, W2s)
			t2 		=	tf.matmul(action_input, W2a)
			output 	=	tf.add( tf.add( t1, t2), b2 )
			output 	=	tf.nn.relu( output )
			output 	=	tf.add( tf.matmul(output, W3), b3 )
		return state_input, action_input, output


	def fit(self, state, action, reward, next_state, done, next_valid=[], lr=[], log=[]):
		# Train the critic
		target_value, Qloss 	=	self.train_critic(state, action, reward, next_state, done, lr[1])
		# Train the actor
		if done.any():
			self.train_actor(state[~done.flatten(),:], lr[0])
		# Update the targets
		self.update_target_networks()
		return target_value, Qloss


	def predict(self, state):
		action 	=	self.act(state)
		return self.session.run(self.critic_target_output, feed_dict={self.critic_target_state_input: state, self.critic_target_action_input: action})


	def train_critic(self, state, action, reward, next_state, done, lr):
		next_action		=	self.session.run(self.actor_target_output, feed_dict={self.actor_target_input:next_state, self.actor_target_training:True})
		predicted_target= 	self.session.run(self.critic_target_output, feed_dict={self.critic_target_state_input: next_state, self.critic_target_action_input:next_action})

		targets	=	[]
		for irew, predt, dn in zip(reward, predicted_target, done):
			if dn:
				targets 	+=	[irew]
			else:
				targets 	+=	[[irew[0] + self.discount_rate * predt[0]]]
		# Update the critic's model
		self.session.run( self.optimize_critic, feed_dict={self.y_input:targets, self.critic_state_input:state, self.critic_action_input:action, self.lr_critic:lr} )
		return targets, self.session.run( self.critic_loss, feed_dict={self.y_input:targets, self.critic_state_input:state, self.critic_action_input:action})


	def train_actor(self, state, lr):
		# Predict actor's next action
		actions_updated =	self.session.run( self.actor_output, feed_dict={self.actor_input:state, self.actor_training:True} )
		# Compute the critic's gradient wrt next action
		gradients 		=	self.session.run( self.critic_grad, feed_dict={self.critic_state_input:state, self.critic_action_input:actions_updated} )
		# Optimize the actor's parameters
		self.session.run(self.optimize_actor, feed_dict={self.actor_input:state, self.actor_critic_grad:gradients[0], self.lr_actor:lr, self.actor_training:True})


	def update_summary(self, rew, qsa, qlss, lr, istep):
		if not self.outputdir is None:
			summary 	=	self.session.run(self.merged_sum, feed_dict={self.lr_actor:lr[0], self.lr_critic:lr[1], self.episode_reward:rew, self.episode_maxQ:qsa, self.episode_Qlss:qlss})
			self.sumWriter.add_summary(summary, istep)


	def update_target_networks(self):
		self.session.run(self.update_actor_target_params)
		self.session.run(self.update_critic_target_params)


	def batch_norm_layer(self, x, training_phase, activation=None, scope_bn=''):
		return tf.cond(training_phase,
					   lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
															updates_collections=None, is_training=True, reuse=None,
															scope=scope_bn, decay=0.9, epsilon=1e-5),
					   lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
															updates_collections=None, is_training=False, reuse=True,
															scope=scope_bn, decay=0.9, epsilon=1e-5))


	def act(self, state, valid_actions=[]):
		return self.session.run( self.actor_output, feed_dict={self.actor_input:state, self.actor_training:True})


	def save_model(self, root, sess, env_name, n_steps):
		# Time stamp
		now 	=	datetime.now()
		stamp 	=	'{}{}{}_{}h{}'.format(now.day, now.month, now.year, now.hour, now.minute)
		svname 	=	path.join(root, 'DDPG_trainedOn_{}_{}_{}steps'.format(env_name, stamp, n_steps))
		self.model_saver.save( sess, svname )
		return svname


	def load_model(self, svname):
		sess	=	tf.Session()
		saver 	= 	tf.train.import_meta_graph( path.join(svname, path.basename(svname) + '.meta'))
		saver.restore(sess, tf.train.latest_checkpoint(path.join(svname, './')) )
		# Link TF variables to the classifier class
		graph 	= 	sess.graph
		annX 	= 	graph.get_tensor_by_name('Input_to_the_network-player_features:0')
		"""self.annY_  =   graph.get_tensor_by_name('Ground_truth:0')
		self.annW1  =   graph.get_tensor_by_name('weights_inp_hid:0')
		self.annB1  =   graph.get_tensor_by_name('bias_inp_hid:0')
		self.Y1     =   graph.get_operation_by_name('hid_output')
		self.annW2  =   graph.get_tensor_by_name('weights_hid_out:0')
		self.annB2  =   graph.get_tensor_by_name('bias_hid_out:0')"""
		annY = graph.get_tensor_by_name('prediction:0')
		return sess, annX, annY


class DQN_agent():

	def __init__(self, state_size, action_size, layer_units, learning_rate, discount_factor, outputdir):
		self.state_size 	= 	state_size
		self.action_size 	= 	action_size
		self.layer_units 	= 	layer_units
		self.discount_factor=	discount_factor
		self.qmodel 		= 	'DQN'
		# Summary variables
		self.sum_epRew 		= 	0
		self.sum_maxQ 		= 	0
		self.sum_totL 		= 	0
		self.learning_rate 	= 	learning_rate
		self.model, self.model_name = self.build_model(state_size, action_size, layer_units, learning_rate, outputdir)


	def build_model(self, state_size, action_size, layer_units, learning_rate, outputdir, activation='relu'):

		def summarize_lr(y_true, y_pred):
			return tf.convert_to_tensor(self.learning_rate, np.float32)
		def summarize_epRew(y_true, y_pred):
			return tf.convert_to_tensor(self.sum_epRew, np.float32)
		def summarize_maxQ(y_true, y_pred):
			return tf.convert_to_tensor(self.sum_maxQ, np.float32)
		def summarize_totL(y_true, y_pred):
			return tf.convert_to_tensor(self.sum_totL, np.float32)

		# if self.wavelet_channels==0:
		# Purely dense MLP
		model = K.models.Sequential()
		model.add(K.layers.Dense(layer_units[0], input_shape=(np.prod(state_size),)))

		for i in range(1, len(layer_units)):
			model.add(K.layers.Dense(layer_units[i], activation=activation))
		# model.add(keras.layers.Dropout(drop_rate))
		model.add(K.layers.Dense(action_size, activation='linear'))
		# --- Prepare summaries
		self.tbcbck 	=	K.callbacks.TensorBoard(log_dir=outputdir, histogram_freq=0, write_graph=True, write_images=True)
		model.compile(loss='mse', optimizer=K.optimizers.Adam(lr=learning_rate), metrics=[summarize_lr, summarize_epRew, summarize_maxQ, summarize_totL])
		return model, 'DQN' + str(layer_units)


	def predict(self, state):
		state 	= 	np.reshape(state, [1,-1])
		q 		= 	self.model.predict(state)
		if np.isnan(np.max(q, axis=1)).any():
			print('state' + str(state))
			print('q' + str(q))
			raise ValueError
		return q.flatten()


	def get_q_valid(self, state, valid_actions):
		#self.model.model.set_weights(self.model.model.get_weights())
		q       =   self.predict(state)
		q_valid =   [np.nan] * len(q)
		for action in valid_actions:
			q_valid[action] =   q[action]
		return q_valid


	def get_targets(self, state, action, reward, next_state, done, next_valid):
		# Compute targets from rewards
		targets = []
		for r, s_, d, n in zip(reward, next_state, done, next_valid):
			q 	= 	r
			if not d:
				try:
					q 	+= 	self.discount_factor * np.nanmax(self.get_q_valid(s_, n))
				except:
					print('pause here')
			targets += 	[q]

		# Format targets
		targetF 	=	[]
		errors 		=	0
		for s, a, t in zip(state, action, targets):
			q 		=	self.predict(s)
			q_ 		=	deepcopy(q)
			q[a]	=	t
			targetF	+=	[q]
			errors 	+=	np.sqrt( np.sum( (q-q_) ** 2 ) )
		return np.array(targetF), errors/len(state)


	def fit(self, state, action, reward, next_state, done, next_valid, lr=[], log=False):
		q, error 	= 	self.get_targets(state, action, reward, next_state, done, next_valid)
		if log:
			self.model.fit(state, q, epochs=1, verbose=0, callbacks=[self.tbcbck])
		else:
			self.model.fit(state, q, epochs=1, verbose=0)
		return error


	def act(self, state, valid_actions):
		q_valid 	= 	self.get_q_valid(state, valid_actions)
		return np.nanargmax(q_valid)


	def update_summary(self, epRew, maxQ, totL, lr, ):
		self.sum_epRew 	=	epRew
		self.sum_maxQ 	=	maxQ
		self.sum_totL 	=	totL



def add_dim(x, shape):
	return np.reshape(x, (1,) + shape)

# ======
# Tester
# ======
def play_episode(env, agent, display=True):
	s 	=	env.reset()
	done=	False
	env.render(s, 0, 0)
	while not done:
		a 	=	agent.act(s)
		s_, r, done, _	=	env.step(a)
		env.render(s, a, r)
		s 	=	s_
	env.kill()


def test_ddpg_gradients():
	# Test whether the agent's action output moves towards the overall gradient

	# Create a fake step
	agent 	=	DDPGModelMLP(10,1)
	s		=	list( range(10) )
	s_ 		=	s[::-1]

	# Test forward prop
	a		=	agent.act(s, [0])
	done, lr=	False, [1e-2, 1e-1]

	# ======
	# Test backprop through the critic
	# ======
	# Get q-value
	q_sa 	= 	agent.session.run(agent.critic_output, feed_dict={agent.critic_state_input: [s], agent.critic_action_input: a})
	# Drive reward up and recompute value
	r		=	q_sa+2
	agent.session.run(agent.optimize_critic, feed_dict={agent.y_input: r, agent.critic_state_input: [s], agent.critic_action_input: a, agent.lr_critic: lr[1]})
	newq1 	=	agent.session.run(agent.critic_output, feed_dict={agent.critic_state_input: [s], agent.critic_action_input: a}).flatten()
	# Drive reward down and recompute value
	r 		= 	newq1-2
	agent.session.run(agent.optimize_critic,feed_dict={agent.y_input: [r], agent.critic_state_input: [s], agent.critic_action_input: a, agent.lr_critic: lr[1]})
	newq2 	= 	agent.session.run(agent.critic_output, feed_dict={agent.critic_state_input: [s], agent.critic_action_input: a}).flatten()
	# Assess gradient direction
	assert(newq1>q_sa and newq1>newq2), 'Critics gradients pointing the wrong direction'

	# ======
	# Test backprop through the whole model
	# ======
	# Get action and value
	a 		=	agent.act(s, [0])
	q_sa 	= 	agent.session.run(agent.critic_output, feed_dict={agent.critic_state_input: [s], agent.critic_action_input: a})
	# ---Drive reward up and recompute action
	r 		= 	q_sa + 2
	# Train critic
	agent.session.run(agent.optimize_critic, feed_dict={agent.y_input: r, agent.critic_state_input: [s], agent.critic_action_input: a, agent.lr_critic: lr[1]})
	# Train actor
	agent.train_actor([s], lr[0])
	a_new 	=	agent.act(s, [0])
	# ---Drive reward up and recompute action
	newq 	=	agent.session.run(agent.critic_output, feed_dict={agent.critic_state_input: [s], agent.critic_action_input: a_new}).flatten()
	r 		= 	newq - 2
	# Train critic
	agent.session.run(agent.optimize_critic, feed_dict={agent.y_input: [r], agent.critic_state_input: [s], agent.critic_action_input: a_new, agent.lr_critic: lr[1]})
	# Train actor
	gradients = agent.session.run(agent.critic_grad, feed_dict={agent.critic_state_input: [s], agent.critic_action_input:a_new})
	# Optimize the actor's parameters
	agent.session.run(agent.optimize_actor, feed_dict={agent.actor_input: [s], agent.actor_critic_grad: gradients[0], agent.lr_actor: lr[0], agent.actor_training: True})
	a_new2 	= 	agent.act(s, [0])
	assert (a_new > a and a_new > a_new2), 'Actor gradients pointing the wrong direction'

	print('Gradient tests complete')


def test_gym_pendulum():
	# Environment variables
	outdir  =   '/home/younesz/Desktop/SUM'

	# import gym environment
	import gym
	env     	=   gym.make('Pendulum-v0')
	agent_opt 	= 	{'type': 'DDPG', 'acSpace': 1, 'lr': [1e-4, 1e-3], 'nzproc':'OrnsteinUhlenbeck', 'batch_size':64, 'discount_factor':0.99}
	agent 		= 	Agent(agent_opt['type'], env.observation_space.shape[0], agent_opt['acSpace'], layer_units=[400, 300], batch_size=agent_opt['batch_size'], noise_process=agent_opt['nzproc'], outputdir=outdir, learning_rate=agent_opt['lr'])

	# Simulation variables
	n_episodes  =   1000
	rewards     =   []
	for episode in range(n_episodes):
		print('episode {}, '.format(episode), end='')
		state   =   np.reshape(env.reset(), [1,-1])
		lr_disc =	agent_opt['lr'] # * n_episodes / (9 + episode + n_episodes)
		epRew   =   0
		epQsa 	=	0
		epQlss 	=	0
		# Train
		for step in range(env.spec.timestep_limit):
			#env.render()
			# Act
			action  =   agent.act(state , 0.05)
			next_state, reward, done, _ =   env.step(action)
			# Store experience
			agent.remember(state, action, reward, next_state.T, done)
			Q_s_a_, Qloss 	= 	agent.replay(learning_rate=lr_disc)
			# Prepare next iteration
			state   =   next_state.T
			epRew   +=  reward[0]
			if len(Q_s_a_)>0:
				epQsa 	=	np.maximum(np.max(Q_s_a_), epQsa) if step>0 else np.max(Q_s_a_)
				epQlss 	+=	np.mean(Qloss)
			if done:
				break
		rewards     +=  [epRew]
		print('total reward %.2f, max Qvalue: %.2f, total loss: %.2f, actorLR: %.4f, criticLR: %.3f, done in %i steps'%(epRew, epQsa, epQlss, lr_disc[0], lr_disc[1], step))

		# Update summary log
		if not episode%1:
			agent.model.update_summary(epRew, 0, 0, lr_disc, episode)

	# Plot result
	F   =   plt.figure()
	Ax  =   F.add_subplot(111)
	Ax.plot(rewards)
	Ax.set_xlabel('Training episode')
	Ax.set_ylabel('Cumulated reward')


def test_gym_pendulum_PEMAMI():
	import gym
	env	=	gym.make('Pendulum-v0')
	np.random.seed(np.random.randint(1e6))
	env.seed(np.random.randint(1e6))

	state_dim	= env.observation_space.shape[0]
	action_dim 	= env.action_space.shape[0]
	action_bound= env.action_space.high
	# Ensure action bound is symmetric
	assert (env.action_space.high == -env.action_space.low)

	agent 	=	Agent('DDPG', state_dim, action_dim, layer_units=[400,300], batch_size=64, discount_factor=0.99, noise_process='OrnsteinUhlenbeck', outputdir='/home/younesz/Desktop/SUM')

	agent.model.update_target_networks()


	for i in range(1000):

		s 		= 	np.reshape(env.reset(), [1,-1])
		Qsa 	=	0
		Qloss	=	0
		epRew 	=	0
		for j in range(100):

			a 	= 	agent.act(s, 0.05)

			s2, r, terminal, info = env.step(a)

			agent.remember(s, a, r, s2.T, terminal)
			qsa, qlss 	=	agent.replay(learning_rate=[1e-4, 1e-3])

			# Update target networks
			agent.model.update_target_networks()

			s 		= 	s2.T
			epRew 	+=	r
			if len(qsa)>0:
				Qsa		+=	qsa
				Qloss 	+=	qlss
			if terminal:
				break

		agent.model.update_summary(epRew[0], Qsa, Qloss, [1e-4, 1e-3], i)
		print('total reward %.2f, max Qvalue: %.2f, total loss: %.2f, actorLR: %.4f, criticLR: %.3f, done in %i steps' % (epRew, Qsa, Qloss, 1e-4, 1e-3, i))


def test_sine():
	# Environment variables
	outdir = 	'/home/younesz/Desktop/SUM'
	exploR =	0.05

	# Agent specific options
	#agent_opt 	=	{'type': 'DQN', 'acSpace':5, 'lr':1e-3, 'nz':'dummy'}
	agent_opt 	= 	{'type': 'DDPG', 'acSpace': 1, 'lr': [1e-4, 1e-3], 'nz':'dummy'}

	# import gym environment
	from generic.environments import Sine
	env 	= 	Sine(action_size=agent_opt['acSpace'])
	agent 	=	Agent( agent_opt['type'], env.observation_space, agent_opt['acSpace'], layer_units=[80,60], noise_process= agent_opt['nz'], outputdir=outdir, learning_rate=agent_opt['lr'])

	# Simulation variables
	n_episodes 	= 	1000
	rewards 	= 	[]
	for episode in range(n_episodes):
		print('episode {}, '.format(episode), end='')
		state 	= 	env.reset()
		lr_disc = 	agent_opt['lr'] #* n_episodes / (9 + episode + n_episodes)
		epRew 	= 	0
		epQlss 	=	0
		epQsa 	= 	0
		# Train
		for step in range(env.spec['timestep_limit']):
			# Act
			action 	= 	agent.act(state, exploR)
			next_state, reward, done, _	=	env.step(action)
			#env.render(state, reward)
			# Store experience
			agent.remember(state, action, reward, next_state, done, list(range(agent_opt['acSpace'])))
			Q_s_a_, Qloss 	=	agent.replay(log=step==0, learning_rate=lr_disc)
			# Prepare next iteration
			state	= 	next_state
			epRew 	+= 	reward.flatten()[0]
			if len(Q_s_a_)>0:
				epQsa 	=	np.maximum(np.max(Q_s_a_), epQsa) if step>0 else np.max(Q_s_a_)
				epQlss 	+=	np.mean(Qloss)
			if done:
				break
		rewards	+=	[epRew]
		print('total reward %.2f, done in %i steps, max value: %.5f' % (epRew, step, epQsa))

		# Update summary log
		agent.model.update_summary(epRew, epQsa, epQlss, lr_disc, episode)


		if not episode % 20:
			print('pause here')
			#play_episode(env, agent, True)


	# Plot result
	F	=	plt.figure()
	Ax	=	F.add_subplot(111)
	Ax.plot(rewards)
	Ax.set_xlabel('Training episode')
	Ax.set_ylabel('Cumulated reward')


def test_sine2():
	# Environment variables
	outdir = '/Users/younes_zerouali/Desktop/SUM'
	lrAcCr = np.array([1e-4, 1e-3])

	# import gym environment
	from generic.environments import Sine
	env 	= 	Sine()
	noise 	=	OrnsteinUhlenbeckActionNoise(mu=np.zeros(env.action_space))
	agent 	=	DDPGModelMLP(env.observation_space, env.action_space, outputdir=outdir,\
							   actor_hidden=[40, 30], critic_hidden=[40,30], noise_process= noise)

	# Simulation variables
	n_episodes 	= 	1000
	rewards 	= 	[]
	epQsa 		=	0
	epQlss 		=	0
	for episode in range(n_episodes):
		print('episode {}, '.format(episode), end='')
		if episode==20:
			print('')
		state 	= 	env.reset()
		lr_disc = 	lrAcCr #* n_episodes / (9 + episode + n_episodes)
		epRew 	= 	0
		epQlss 	=	0
		# Train
		for step in range(env.spec['timestep_limit']):
			if step==35:
				print('')
			action	=	agent.session.run(agent.actor_output, feed_dict={agent.actor_input:[state], agent.actor_training:True})[0][0] + agent.noise_process()[0]
			next_state, reward, done, _	= env.step(action)
			epRew 	+=	reward
			agent.remember(state, action, reward, next_state, done)

			# Keep adding experience to the memory until
			# there are at least minibatch size samples
			if len(agent.memory) > agent.batch_size:
				MEM	=	list(compress(agent.memory, np.random.randint(len(agent.memory), size=agent.batch_size)))
				s_batch 	=	[x[0] for x in MEM]
				a_batch 	=	[[x[1]] for x in MEM]
				r_batch 	=	[x[2] for x in MEM]
				s2_batch 	=	[x[3] for x in MEM]
				t_batch 	=	[x[4] for x in MEM]

				# Calculate targets
				pred_a 		=	agent.session.run( agent.actor_target_output, feed_dict={agent.actor_target_input:s_batch, agent.actor_target_training:True})
				target_q 	= 	agent.session.run( agent.critic_target_output, feed_dict={agent.critic_target_state_input:s2_batch, agent.critic_target_action_input:pred_a})

				y_i = []
				for k in range(len(MEM)):
					if t_batch[k]:
						y_i.append(np.array([r_batch[k]]))
					else:
						y_i.append(r_batch[k] + agent.discount_rate * target_q[k])

				# Update the critic given the targets
				agent.session.run(agent.optimize_critic, feed_dict={agent.y_input: y_i, agent.critic_state_input: s_batch, agent.critic_action_input: a_batch, agent.lr_critic: lrAcCr[1]})

				# Update the actor policy using the sampled gradient
				a_outs 	= 	agent.session.run(agent.actor_output, feed_dict={agent.actor_input:s_batch, agent.actor_training:True})
				grads 	=	agent.session.run(agent.critic_grad, feed_dict={agent.critic_state_input:s_batch, agent.critic_action_input:a_batch})
				agent.session.run(agent.optimize_actor, feed_dict={agent.actor_input: s_batch, agent.actor_critic_grad: grads[0], agent.actor_training:True, agent.lr_actor:lrAcCr[0]})

				# Update target networks
				agent.update_target_networks()

			state	=	next_state
		print('total reward %.2f, done in %i steps, max value: %.5f' % (epRew, step, epQsa))


	# Plot result
	F	=	plt.figure()
	Ax	=	F.add_subplot(111)
	Ax.plot(rewards)
	Ax.set_xlabel('Training episode')
	Ax.set_ylabel('Cumulated reward')


def test_market():

	# Environment variables
	outdir 			= 	'/home/younesz/Desktop/SUM'
	window_state	=	32
	open_cost 		=	3
	time_difference =	True
	wavelet_channels=	0
	batch_size 		=	16
	rootStore 		= 	open('../dbloc.txt', 'r').readline().rstrip('\n')

	# Agent specific options
	#agent_opt 	=	{'type': 'DQN', 'acSpace':5, 'lr':1e-3, 'nz':'dummy', 'batch_size':batch_size, 'action_labels':['short100', 'short50', 'neutral', 'long50' ,'long100']}
	agent_opt 	= 	{'type': 'DDPG', 'acSpace': 1, 'lr': [1e-4, 1e-3], 'nz': 'dummy', 'batch_size':batch_size, 'action_labels':['continuous'], 'action_conversion':None}

	# import market environment
	from src.emulator import Market
	from src.sampler import SinSampler, BTCsampler
	#sampler 	= 	SinSampler('single', 180, 1.5, (20, 40), (49, 50), fld=outdir)
	sampler 	= 	BTCsampler(False, wavelet_channels=0, variables=['Close'], fld= path.join(rootStore, 'data', 'BTCsampler', 'db_bitcoin.csv'))
	env 		= 	Market(sampler, window_state, open_cost, time_difference=time_difference, wavelet_channels=wavelet_channels, action_range=[-1,1], action_labels=agent_opt['action_labels'])

	# Set agent
	agent 		= 	Agent(agent_opt['type'], window_state, agent_opt['acSpace'], layer_units=[80, 60],
				  			noise_process=agent_opt['nz'], outputdir=outdir, learning_rate=agent_opt['lr'],
							batch_size=agent_opt['batch_size'])
	agent.p_model= 	agent.model
	fld_save 	= 	path.join(rootStore, 'results', sampler.title, agent_opt['type'],
							str((env.window_state, sampler.window_episode, batch_size, agent_opt['lr'],
								 agent.discount_factor, 0, env.open_cost)))


	# Set visualizer
	from src.visualizer import Visualizer
	visualizer 	= 	Visualizer(env.action_labels)

	# Set simulator
	from src.simulators import Simulator
	simulator 	= 	Simulator(agent, env, visualizer=visualizer, fld_save=fld_save)
	simulator.agent_opt 	=	agent_opt
	# Train
	simulator.train(200, save_per_episode=1, exploration_decay=0.99, learning_rate=agent_opt['lr'],
					exploration_min=0.05, print_t=False, exploration_init=0.8)
	# Test
	simulator.test(50, save_per_episode=1, subfld='in-sample testing')



if __name__ == '__main__':
	# Launch gradients test
	#test_ddpg_gradients()

	# Launch gym training
	#test_gym_pendulum()
	#test_gym_pendulum_PEMAMI()

	# Launch sine wave training
	#test_sine()

	# Launch market testing
	test_market()
