import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from os import path
from datetime import datetime
from itertools import compress


class DDPGModelMLP():
	# Actor-critic method, both are implemented as MLPs

	def __init__(self, state_space, action_space, \
				 exploration_rate	=	0.1,\
				 buffer_size		=	200,\
				 batch_size			=	32,\
				 cache_size 		=	1000,\
				 tau				=	0.001,\
				 discount 			=	0.8,\
				 outputdir			=	None,\
				 actor_hidden		=	[400, 300],\
				 critic_hidden		=	[400, 300]):
		self.qmodel             =   'MLP_DDPG'
		self.session            =   tf.Session()
		self.state_space        =   state_space
		self.action_space       =   action_space
		self.exploration_rate 	=	exploration_rate
		self.buffer_size 		=	buffer_size
		self.batch_size 		=	batch_size
		self.cache_size 		=	cache_size
		self.update_rate 		=	tau
		self.discount_rate 		=	discount
		self.random_seed 		=	np.random.randint(1e9, size=6)
		self.memory 			=	[]
		self.model_saver 		=	None
		self.outputdir 			=	outputdir
		# Make the graph
		self.actor_hidden       =   actor_hidden
		self.critic_hidden      =   critic_hidden
		self.build_graph()


	def build_graph(self):
		# --- Setup actor
		# Build networks
		self.actor_weights_handles 			= 	['ac_w1', 'ac_b1', 'ac_w2', 'ac_b2', 'ac_w3', 'ac_b3']
		self.actor_input, self.actor_output, self.actor_training						=	self.build_actor(self.state_space, self.actor_hidden, self.action_space)
		self.actor_target_input, self.actor_target_output, self.actor_target_training	=	self.build_actor(self.state_space, self.actor_hidden, self.action_space, 'target')
		# Build graphs ops - actor
		self.actor_critic_grad 	=	tf.placeholder(tf.float32, [None, self.action_space], name="dQ_dCritic")
		actor_model_weights 	=	self.get_weights('ac', parent_scope='actor_')
		self.actor_grad			=	tf.gradients(self.actor_output, actor_model_weights, -self.actor_critic_grad, name="dAction_dActorParams")
		grads 					=	zip(self.actor_grad, actor_model_weights)
		# Set actor's learning rate
		with tf.name_scope('actor_lr'):
			self.lr_actor 		=	tf.placeholder(tf.float32)
			tf.summary.scalar('actor_learning_rate', self.lr_actor)
		self.optimize_actor	=	tf.train.AdamOptimizer(self.lr_actor).apply_gradients(grads)

		# --- Setup critic
		self.critic_state_input, self.critic_action_input, self.critic_output	=	self.build_critic(self.state_space, self.critic_hidden, self.action_space)
		self.critic_target_state_input, self.critic_target_action_input, self.critic_target_output	= 	self.build_critic(self.state_space, self.critic_hidden, self.action_space, 'target')
		# Set critic's learning rate
		with tf.name_scope('critic_lr'):
			self.lr_critic		=	tf.placeholder(tf.float32)
			tf.summary.scalar('critic_learning_rate', self.lr_critic)
		# Build graphs ops - critic
		self.critic_grad 	= 	tf.gradients(self.critic_output, self.critic_action_input, name="dQ_dAction")
		self.y_input 		=	tf.placeholder("float", shape=[None, self.action_space], name="Qvalue_target")
		critic_loss 		=	tf.reduce_mean( tf.square(self.y_input - self.critic_output), name="critic_loss" )
		self.optimize_critic=	tf.train.AdamOptimizer(self.lr_critic).minimize(critic_loss)

		# Add reward to graph summary
		with tf.name_scope('episode_total_reward'):
			self.episode_reward	=	tf.placeholder(tf.float32)
			tf.summary.scalar('episode_reward', self.episode_reward)
		self.merged_sum 	= 	tf.summary.merge_all()

		# --- Initialize graph
		if not self.outputdir is None:
			self.sumWriter 	=	tf.summary.FileWriter(self.outputdir, self.session.graph)
		self.model_saver	=	tf.train.Saver()
		self.session.run(tf.initialize_all_variables())

		# Prepare update operation
		self.update_network_weights = self.update_target_networks()


	def build_actor(self, input_size, actor_hidden, output_size, handle_suffix=''):
		with tf.name_scope('actor_'+handle_suffix):
			# Input layer
			state_input	=	tf.placeholder("float", [None, input_size], name="actor_state_input")
			is_training	=	tf.placeholder("bool", name="actor_training_flag")

			# --- VARIABLES
			# Hidden layer 1
			W1 	=	tf.Variable( tf.random_uniform([input_size, actor_hidden[0]],-3e-3,3e-3, seed=self.random_seed[0]), name= 'ac_w1'+handle_suffix )
			b1 	=	tf.Variable( tf.random_uniform([actor_hidden[0]],-3e-3,3e-3, seed=self.random_seed[1]), name='ac_b1'+handle_suffix )
			# Hidden layer 2
			W2 	= 	tf.Variable( tf.random_uniform([actor_hidden[0], actor_hidden[1]], -3e-3, 3e-3, seed=self.random_seed[2]), name= 'ac_w2'+handle_suffix )
			b2 	= 	tf.Variable( tf.random_uniform([actor_hidden[1]], -3e-3, 3e-3, seed=self.random_seed[3]), name='ac_b2'+handle_suffix )
			# Hidden layer 3
			W3 	= 	tf.Variable( tf.random_uniform([actor_hidden[1], output_size], -3e-3,3e-3, seed=self.random_seed[4]), name= 'ac_w3'+handle_suffix )
			b3 	= 	tf.Variable( tf.random_uniform([output_size], -3e-3, 3e-3, seed=self.random_seed[5]), name='ac_b3'+handle_suffix )

			# --- GRAPH
			output 	=	state_input #self.batch_norm_layer(state_input, is_training, activation=tf.identity, scope_bn='batch_norm0'+handle_suffix)
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
			W1 	=	tf.Variable( tf.random_uniform([input_size, critic_hidden[0]], -3e-3, 3e-3, seed=self.random_seed[0]), name= 'cr_w1'+handle_suffix )
			b1 	=	tf.Variable( tf.random_uniform([critic_hidden[0]], -3e-3, 3e-3, seed=self.random_seed[1]), name= 'cr_b1'+handle_suffix )
			# Hidden layer 2: take up concatenation on action_input
			W2 	=	tf.Variable( tf.random_uniform([critic_hidden[0]+output_size, critic_hidden[1]], -3e-3, 3e-3, seed=self.random_seed[2]), name= 'cr_w2'+handle_suffix  )
			b2 	=	tf.Variable( tf.random_uniform([critic_hidden[1]], -3e-3, 3e-3, seed=self.random_seed[3]), name= 'cr_b2'+handle_suffix )
			# Output layer
			W3 	=	tf.Variable( tf.random_uniform([critic_hidden[1], output_size], -3e-3, 3e-3, seed=self.random_seed[4]), name= 'cr_w3'+handle_suffix  )
			b3 	=	tf.Variable( tf.random_uniform([output_size], -3e-3, 3e-3, seed=self.random_seed[5]), name= 'cr_b3'+handle_suffix )

			# --- GRAPH
			output 	=	tf.add( tf.matmul(state_input, W1), b1 )
			output 	=	tf.nn.relu( output )
			output 	=	tf.concat( [output, action_input], 1 )
			output 	=	tf.add( tf.matmul(output, W2), b2 )
			output 	=	tf.nn.relu( output )
			output 	=	tf.add( tf.matmul(output, W3), b3 )
		return state_input, action_input, output


	def get_weights(self, component, suffix='', parent_scope=''):
		w1 	=	tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=path.join(parent_scope, component + '_w1' + suffix + ':0') )[0]
		b1 	= 	tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=path.join(parent_scope, component + '_b1' + suffix + ':0') )[0]
		w2 	= 	tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=path.join(parent_scope, component + '_w2' + suffix + ':0') )[0]
		b2 	= 	tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=path.join(parent_scope, component + '_b2' + suffix + ':0') )[0]
		w3 	=	tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=path.join(parent_scope, component + '_w3' + suffix + ':0') )[0]
		b3 	= 	tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=path.join(parent_scope, component + '_b3' + suffix + ':0') )[0]
		return [w1, b1, w2, b2, w3, b3]


	def fit(self, state, action, reward, next_state, done, lr):
		# Train the critic
		self.train_critic(state, action, reward, next_state, done, lr[1])
		# Train the actor
		if not done:
			self.train_actor(state, lr[0])
		# Update the targets
		self.session.run(self.update_network_weights)


	def train_critic(self, state, action, reward, next_state, done, lr):
		if not done:
			# Compute next action
			next_action	=	self.session.run(self.actor_target_output, feed_dict={self.actor_target_input:next_state, self.actor_target_training:True})
			# Compute future reward *** CAN IMPLEMENT TD(lambda) HERE
			reward 		+=	self.discount_rate * self.session.run(self.critic_target_output, feed_dict={self.critic_target_state_input:next_state, self.critic_target_action_input:next_action})[0]
		# Update the critic's model
		self.session.run( self.optimize_critic, feed_dict={self.y_input:reward, self.critic_state_input:state, self.critic_action_input:action, self.lr_critic:lr} )


	def train_actor(self, state, lr):
		# Predict actor's next action
		actions_updated =	self.session.run( self.actor_output, feed_dict={self.actor_input:state, self.actor_training:True} )
		# Compute the critic's gradient wrt next action
		gradients 		=	self.session.run( self.critic_grad, feed_dict={self.critic_state_input:state, self.critic_action_input:actions_updated} )
		# Optimize the actor's parameters
		self.session.run(self.optimize_actor, feed_dict={self.actor_input:state, self.actor_critic_grad:gradients[0], self.lr_actor:lr, self.actor_training:True})


	def update_summary(self, lr, rew, istep):
		summary 	=	self.session.run(self.merged_sum, feed_dict={self.lr_actor:lr[0], self.lr_critic:lr[1], self.episode_reward:rew[0]})
		self.sumWriter.add_summary(summary, istep)


	def update_target_networks(self):
		# Get weights - critic
		critic_model_ref 	=	self.get_weights('cr', parent_scope='critic_')
		critic_target_ref 	=	self.get_weights('cr', 'target', parent_scope='critic_target')
		critic_target_val 	= 	[]
		# Get weights - actor
		actor_model_ref 	= 	self.get_weights('ac', parent_scope='actor_')
		actor_target_ref 	= 	self.get_weights('ac', 'target', parent_scope='actor_target')
		actor_target_val 	= 	[]
		for iw in range( len(critic_target_ref) ):
			critic_target_val	+=	[(1-self.update_rate) * critic_target_ref[iw] + self.update_rate * critic_model_ref[iw]]
			actor_target_val 	+= 	[(1-self.update_rate) * actor_target_ref[iw] + (1-self.update_rate) * actor_model_ref[iw]]
		# Update
		upd_critic 	=	[tf.assign(x, y) for x,y in zip(critic_target_ref, critic_target_val)]
		upd_actor 	=	[tf.assign(x, y) for x,y in zip(actor_target_ref, actor_target_val)]
		return [upd_actor, upd_critic]


	def batch_norm_layer(self, x, training_phase, activation=None, scope_bn=''):
		return tf.cond(training_phase,
					   lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
															updates_collections=None, is_training=True, reuse=None,
															scope=scope_bn, decay=0.9, epsilon=1e-5),
					   lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
															updates_collections=None, is_training=False, reuse=True,
															scope=scope_bn, decay=0.9, epsilon=1e-5))


	def act(self, state, default):
		if np.random.random()<self.exploration_rate:
			return default
		else:
			return self.session.run( self.actor_output, feed_dict={self.actor_input:[state], self.actor_training:True})


	def remember(self, state, action, reward, next_state, done):
		if  len(self.memory)==self.cache_size:
			self.memory.pop(0)
		self.memory.append( (state, action, reward, next_state, done) )


	def replay(self, lr):
		if len(self.memory)>self.buffer_size:
			# Sample batch
			state_btch 	=	[]
			action_btch =	[]
			reward_btch =	[]
			nextSt_btch = 	[]
			done_btch 	=	[]
			for (state, action, reward, next_state, done) in compress(self.memory, np.random.randint(len(self.memory), size=self.batch_size)):
				state_btch 	+=	[list(state)]
				action_btch +=	[[action[0][0]]]
				reward_btch +=	[[reward[0]]]
				nextSt_btch += 	[list(next_state[0])]
				done_btch	+=	[done]
			self.fit(state_btch, action_btch, reward_btch, nextSt_btch, done_btch, lr)


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



# ======
# Tester
# ======
def test_ddpg_gradients():
	# Test whether the agent's action output moves towards the overall gradient

	# Create a fake step
	agent 	=	DDPGModelMLP(10,1)
	s		=	list( range(10) )
	s_ 		=	s[::-1]


	# Test forward prop
	ac_ref 	= 	agent.get_weights('ac', parent_scope='actor_')
	ac_val 	=	[agent.session.run(x) for x in ac_ref]

	a		=	agent.act(s, [0])
	done, lr=	False, [1e-2, 1e-3]

	# Test backprop through the critic
	q_sa 	= 	agent.session.run(agent.critic_output, feed_dict={agent.critic_state_input: [s], agent.critic_action_input: a})
	r		=	2 + q_sa



	agent.session.run(agent.optimize_critic, feed_dict={agent.y_input: r, agent.critic_state_input: [s], agent.critic_action_input: a, agent.lr_critic: lr[1]})

	# Learn from that step
	agent.fit([s], a, r, [s_], done, lr)
	print('Original action: %.5f, Incentivized action: %.5f'%(a, agent.act(s,0)))


def test_gym_pendulum():
	# Environment variables
	outdir  =   '/home/younesz/Desktop/SUM'
	lrAcCr 	=	np.array([1e-4, 1e-3])

	# import gym environment
	import gym
	env     =   gym.make('Pendulum-v0')
	agent   =   DDPGModelMLP( env.observation_space.shape[0], env.action_space.shape[0], outputdir=outdir )

	# Simulation variables
	n_episodes  =   1000
	rewards     =   []
	for episode in range(n_episodes):
		print('episode {}, '.format(episode), end='')
		state   =   env.reset()
		lr_disc =	lrAcCr * n_episodes / (9 + episode + n_episodes)
		epRew   =   0
		# Train
		for step in range(env.spec.timestep_limit):
			#env.render()
			# Act
			action  =   agent.act(state , [[np.random.random()*2-1]])
			next_state, reward, done, _ =   env.step(action)
			# Store experience
			agent.remember(state, action, reward, next_state.T, done)
			agent.replay(lr_disc)
			# Prepare next iteration
			state   =   next_state.flatten()
			epRew   +=  reward
			if done:
				break
		rewards     +=  [epRew]
		print('total reward %.2f, done in %i steps'%(epRew, step))

		# Update summary log
		if not episode%20:
			agent.update_summary(lr_disc, epRew, episode)

	# Plot result
	F   =   plt.figure()
	Ax  =   F.add_subplot(111)
	Ax.plot(rewards)
	Ax.set_xlabel('Training episode')
	Ax.set_ylabel('Cumulated reward')



if __name__ == '__main__':
	# Launch gradients test
	test_ddpg_gradients()

	# Launch gym training
	#test_gym_pendulum()
