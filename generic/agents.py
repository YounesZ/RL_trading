import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from os import path
from datetime import datetime


class DDPGModelMLP():
	# Actor-critic method, both are implemented as MLPs

	def __init__(self, state_space, action_space, exploration_rate=0.1, buffer_size=200, batch_size=32, tau=0.8, outputdir=None, actor_hidden=[64, 64], critic_hidden=[64, 64]):
		self.qmodel             =   'MLP_DDPG'
		self.session            =   tf.Session()
		self.state_space        =   state_space
		self.action_space       =   action_space
		self.exploration_rate 	=	exploration_rate
		self.buffer_size 		=	buffer_size
		self.batch_size 		=	batch_size
		self.update_rate 		=	tau
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
		self.actor_input, self.actor_output	=	self.build_actor(self.state_space, self.actor_hidden, self.action_space)
		_, self.actor_target_output			=	self.build_actor(self.state_space, self.actor_hidden, self.action_space, 'target')
		# Build graphs ops - actor
		self.actor_critic_grad 	=	tf.placeholder(tf.float32, [None, self.action_space])
		actor_model_weights 	=	self.get_weights('ac')
		self.actor_grad			=	tf.gradients(self.actor_output, actor_model_weights, -self.actor_critic_grad)
		grads 					=	zip(self.actor_grad, actor_model_weights)
		# Set actor's learning rate
		with tf.name_scope('actor_lr'):
			lr_actor 		=	tf.placeholder(tf.float32)
			tf.summary.scalar('actor_learning_rate', lr_actor)
		self.optimize_actor	=	tf.train.AdamOptimizer(lr_actor).apply_gradients(grads)

		# --- Setup critic
		self.critic_state_input, self.critic_action_input, self.critic_output	=	self.build_critic(self.state_space, self.critic_hidden, self.action_space)
		_, _, self.target_critic_ouput	= 	self.build_critic(self.state_space, self.critic_hidden, self.action_space, 'target')
		# Set critic's learning rate
		with tf.name_scope('critic_lr'):
			lr_critic		=	tf.placeholder(tf.float32)
			tf.summary.scalar('critic_learning_rate', lr_critic)
		# Build graphs ops - critic
		self.critic_grad 	= 	tf.gradients(self.critic_output, self.critic_action_input)
		self.y_input 		=	tf.placeholder("float", shape=[self.action_space])
		critic_loss 		=	tf.reduce_mean( tf.square(self.y_input - self.critic_output) )
		self.optimize_critic=	tf.train.AdamOptimizer(lr_critic).minimize(critic_loss)

		# --- Initialize graph
		self.merged_sum 	= 	tf.summary.merge_all()
		if not self.outputdir is None:
			self.sumWriter 	=	tf.summary.FileWriter(self.outputdir, self.session.graph)
		self.model_saver	=	tf.train.Saver()
		self.session.run(tf.initialize_all_variables())


	def build_actor(self, input_size, actor_hidden, output_size, handle_suffix=''):
		# Input layer
		state_input	=	tf.placeholder("float", [None, input_size])

		# --- VARIABLES
		# Hidden layer 1
		W1 	=	tf.Variable( tf.random_uniform([input_size, actor_hidden[0]],-3e-3,3e-3), name= 'ac_w1'+handle_suffix )
		b1 	=	tf.Variable( tf.random_uniform([actor_hidden[0]],-3e-3,3e-3), name='ac_b1'+handle_suffix )
		# Hidden layer 2
		W2 	= 	tf.Variable( tf.random_uniform([actor_hidden[0], actor_hidden[1]], -3e-3, 3e-3), name= 'ac_w2'+handle_suffix )
		b2 	= 	tf.Variable( tf.random_uniform([actor_hidden[1]], -3e-3, 3e-3), name='ac_b2'+handle_suffix )
		# Hidden layer 3
		W3 	= 	tf.Variable( tf.random_uniform([actor_hidden[1], output_size]), name= 'ac_w3'+handle_suffix )
		b3 	= 	tf.Variable( tf.random_uniform([output_size], -3e-3, 3e-3), name='ac_b3'+handle_suffix )

		# --- GRAPH
		output 	=	tf.add( tf.matmul(state_input, W1), b1 )
		output 	=	tf.nn.relu( output )
		output 	=	tf.add( tf.matmul(output, W2), b2 )
		output 	=	tf.nn.relu(output)
		output 	=	tf.add( tf.matmul(output, W3), b3 )
		output 	=	tf.nn.tanh( output )
		return state_input, output


	def build_critic(self, input_size, critic_hidden, output_size, handle_suffix=''):
		# Input layer 1: the state space
		state_input = 	tf.placeholder("float", shape=[None, input_size])
		# Input layer 2: the action
		action_input=	tf.placeholder("float", shape=[output_size])

		# --- VARIABLES
		# Hidden layer 1
		W1 	=	tf.Variable( tf.random_uniform([input_size, critic_hidden[0]]), name= 'cr_w1'+handle_suffix )
		b1 	=	tf.Variable( tf.random_uniform([critic_hidden[0]]), name= 'cr_b1'+handle_suffix )
		# Hidden layer 2: take up concatenation on action_input
		W2 	=	tf.Variable( tf.random_uniform([critic_hidden[0]+output_size, critic_hidden[1]]), name= 'cr_w2'+handle_suffix  )
		b2 	=	tf.Variable( tf.random_uniform([critic_hidden[1]]), name= 'cr_b2'+handle_suffix )
		# Output layer
		W3 	=	tf.Variable( tf.random_uniform([critic_hidden[1], output_size]), name= 'cr_w3'+handle_suffix  )
		b3 	=	tf.Variable( tf.random_uniform([output_size]), name= 'cr_b3'+handle_suffix )

		# --- GRAPH
		output 	=	tf.add( tf.matmul(state_input, W1), b1 )
		output 	=	tf.nn.relu( output )
		output 	=	tf.concat( [output, action_input], 0 )
		output 	=	tf.add( tf.matmul(output, W2), b2 )
		output 	=	tf.nn.relu( output )
		output 	=	tf.add( tf.matmul(output, W3), b3 )
		return state_input, action_input, output


	def get_weights(self, component, suffix=''):
		w1 	=	tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=component + '_w1' + suffix + ':0')[0]
		b1 	= 	tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=component + '_b1' + suffix + ':0')[0]
		w2 	= 	tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=component + '_w2' + suffix + ':0')[0]
		b2 	= 	tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=component + '_b2' + suffix + ':0')[0]
		w3 	=	tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=component + '_w3' + suffix + ':0')[0]
		b3 	= 	tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=component + '_b3' + suffix + ':0')[0]
		return [w1, b1, w2, b2, w3, b3]


	def fit(self, state, action, reward, next_state, done):
		# Train the critic
		self.train_critic(state, action, reward, next_state, done)
		# Train the actor
		if not done:
			self.train_actor(state, next_state)
		# Update the targets
		self.update_target_networks()


	def train_critic(self, state, action, reward, next_state, done):
		if not done:
			# Compute next action
			next_action	=	self.session.run(self.actor_target_output, feed_dict={self.actor_input:next_state})
			# Compute future reward *** CAN IMPLEMENT TD(lambda) HERE
			reward 		+=	self.session.run(self.critic_output, feed_dict={self.critic_state_input:next_state, self.critic_action_input:next_action})
		# Update the critic's model
		self.session.run( self.optimize_critic, feed_dict={self.y_input:reward, self.critic_state_input:state, self.critic_action_input:action} )


	def train_actor(self, state, next_state):
		# Predict actor's next action
		next_action =	self.session.run( self.actor_output, feed_dict={self.actor_input:next_state} )
		# Compute the critic's gradient wrt next action
		gradients 	=	self.session.run( self.critic_grad, feed_dict={self.critic_state_input:next_state, self.critic_action_input:next_action})
		# Optimize the actor's paramters
		if not self.outputdir is None:
			summary, _ 	=	self.session.run([self.merged_sum, self.optimize_actor], feed_dict={self.actor_input:state, self.actor_critic_grad:gradients[0]})
			self.sumWriter.add_summary(summary)
		else:
			self.session.run(self.optimize_actor, feed_dict={self.actor_input: state, self.actor_critic_grad: gradients[0]})


	def update_target_networks(self):
		# Get weights - critic
		critic_model_ref 	=	self.get_weights('ac')
		critic_target_ref 	=	self.get_weights('ac', 'target')
		critic_target_val 	= 	[]
		# Get weights - actor
		actor_model_ref 	= 	self.get_weights('cr')
		actor_target_ref 	= 	self.get_weights('cr', 'target')
		actor_target_val 	= 	[]
		for iw in range( len(critic_target_ref) ):
			critic_target_val	+=	[self.update_rate * self.session.run(critic_target_ref[iw]) + (1-self.update_rate) * self.session.run(critic_model_ref[iw])]
			actor_target_val 	+= 	[self.update_rate * self.session.run(actor_target_ref[iw]) + (1-self.update_rate) * self.session.run(actor_model_ref[iw])]
		# Update
		[self.session.run( tf.assign(x, y) ) for x,y in zip(critic_target_ref, critic_target_val)]
		[self.session.run( tf.assign(x, y) ) for x,y in zip(actor_target_ref, actor_target_val)]


	def act(self, state, default):
		if np.random.random()<self.exploration_rate:
			return default
		else:
			return self.session.run( self.actor_target_output, feed_dict={self.actor_input:state})


	def remember(self, state, action, reward, next_state, done):
		self.memory.append( (state, action, reward, next_state, done) )


	def replay(self):
		if len(self.memory)>self.buffer_size:
			# Sample batch
			for (state, action, reward, next_state, done) in [self.memory[np.random.randint(len(self.memory))]]:
				self.fit(state, action, reward, next_state, done)


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
if __name__ == '__main__':
	# Environment variables
	outdir  =   '/home/younesz/Desktop/SUM'

	# import gym environment
	import gym
	env     =   gym.make('Pendulum-v0')
	agent   =   DDPGModelMLP( env.observation_space.shape[0], env.action_space.shape[0] )

	# Simulation variables
	n_episodes  =   1000
	rewards     =   []
	for episode in range(n_episodes):
		state   =   env.reset()
		epRew   =   0
		# Train
		for step in range(env.spec.timestep_limit):
			env.render()
			# Act
			action  =   agent.act(state , np.random.random()*2-1)
			next_state, reward, done, _ =   env.step(action)
			# Store experience
			agent.remember(state, action, reward, next_state, done)
			agent.replay()
			# Prepare next iteration
			state   =   next_state
			epRew   +=  reward
			if done:
				break
		rewards     +=  [epRew]

	# Plot result
	F   =   plt.figure()
	Ax  =   F.add_subplot(111)
	Ax.plot(rewards)
	Ax.set_xlabel('Training episode')
	Ax.set_ylabel('Cumulatede reward')