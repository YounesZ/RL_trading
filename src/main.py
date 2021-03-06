#!/usr/bin/env python2

from src.lib import *
from src.sampler import *
from src.agents import *
from src.emulator import *
from src.simulators import *
from src.visualizer import *


def get_agent(model_type, env, learning_rate=0.01, fld_load=True, input_size=32, batch_size=8, discount_factor=0.99, buffer_size=2000, buffer_min_size=100, planning_horizon=10, outputdir='/home/younesz/Desktop/SUM'):
	if model_type=='DDPG':
		agent,print_t 	=	get_model(model_type, env, learning_rate=learning_rate, fld_load=fld_load, batch_size=batch_size, buffer_min_size=buffer_min_size, gamma=discount_factor, buffer_size=buffer_size, use_noise=False, outputdir=outputdir)
	else:
		model, print_t 	= 	get_model(model_type, env, learning_rate, fld_load, input_size, batch_size)
		p_model, _ 		=	get_model(model_type, env, learning_rate, fld_load, input_size, batch_size=1, outputdir=outputdir)
		agent			=	Agent(model, discount_factor=discount_factor, batch_size=batch_size,
				  			prediction_model=p_model, buffer_size=buffer_size, planning_horizon=planning_horizon)
	return agent, print_t


def get_model(model_type, env, learning_rate, fld_load, input_size=32, batch_size=8, buffer_size=2000, buffer_min_size=100, gamma=0.99, use_noise=True, outputdir=None):

	print_t = False
	exploration_init = 1.

	if model_type == 'MLP':
		#m = 16
		layers 		= 	5
		layer_units = 	[48, 24, 12, 6] 	#[m]*layers
		model 		= 	QModelMLP(env.state_shape, env.n_action, env.wavelet_channels)
		model.build_model(layer_units, learning_rate=learning_rate, activation='tanh')


	elif model_type == 'MLP_PG':
		# Policy gradient approximation
		layers 		=	5
		hidden_size = 	[48, 24, 12, 6]
		model 		=	PGModelMLP(env.state_shape, env.n_action, env.wavelet_channels)
		model.build_model(hidden_size, learning_rate=learning_rate, activation='relu', input_size=input_size, batch_size=batch_size)


	elif model_type == 'DDPG':
		# Deep Deterministic Policy Gradient
		# Lillicrap et al. 2016: Continuous control with deep reinforcement learning
		from external_modules.patemami_DDPG.ddpg.ddpg_monday import DDPG
		from generic.dummy import Model
		actor_hidden 	=	[48, 24, 12, 6]
		critic_hidden 	=	[[48, 24], [24, 12, 6]]
		model 			=	DDPG(env.state_shape[1], env.n_action, use_noise=use_noise, outputdir=outputdir, batch_size=batch_size, buffer_size=buffer_size, buffer_min_size=buffer_min_size, gamma=gamma)
		model.model 	=	Model('DDPG')
		#model.build_model(actor_hidden, critic_hidden, learning_rate=learning_rate, activation='relu', input_size=input_size)


	elif model_type == 'conv':

		m = 16
		layers = 2
		filter_num = [m]*layers
		filter_size = [3] * len(filter_num)
		#use_pool = [False, True, False, True]
		#use_pool = [False, False, True, False, False, True]
		use_pool = None
		#dilation = [1,2,4,8]
		dilation = None
		dense_units = [48,24]
		model = QModelConv(env.state_shape, env.n_action)
		model.build_model(filter_num, filter_size, dense_units, learning_rate, 
			dilation=dilation, use_pool=use_pool)


	elif model_type == 'RNN':

		m = 32
		layers = 3
		hidden_size = [m]*layers
		dense_units = [m,m]
		model = QModelGRU(env.state_shape, env.n_action)
		model.build_model(hidden_size, dense_units, learning_rate=learning_rate)
		print_t = True


	elif model_type == 'ConvRNN':
	
		m = 8
		conv_n_hidden = [m,m]
		RNN_n_hidden = [m,m]
		dense_units = [m,m]
		model = QModelConvGRU(env.state_shape, env.n_action)
		model.build_model(conv_n_hidden, RNN_n_hidden, dense_units, learning_rate=learning_rate)
		print_t = True


	elif model_type == 'pretrained':
		agent.model = load_model(fld_load, learning_rate)

	else:
		raise ValueError
		
	return model, print_t


def main():

	"""
	it is recommended to generate database using sampler.py before run main
	"""

	# --- Agent's options
	batch_size 		= 	64
	learning_rate 	= 	1e-4
	discount_factor = 	0.8
	exploration_decay= 	0.99
	exploration_min = 	0.01
	buffer_min_size =	100
	buffer_size		=	2000
	planning_horizon=	1

	# DQN architecture
	model_type		=	'DDPG';
	exploration_init= 	1.;
	fld_load 		= 	None

	# --- Environment's options
	rootStore 		=	open('../dbloc.txt', 'r').readline().rstrip('\n')
	window_state 	= 	32
	time_difference = 	True
	wavelet_channels=	0
	n_episode_training= 100
	n_episode_testing = 50
	open_cost 		= 	3  # Percentage on the buy order
	db_type 		= 	'SinSamplerDB'; db = 'concat_half_base_'; Sampler = 	SinSampler
	# db_type = 'PairSamplerDB'; db = 'randjump_100,1(10, 30)[]_'; Sampler = PairSampler


	fld 	= 	os.path.join(rootStore,'data',db_type,db+'A')
	#sampler = Sampler('load', fld=fld)
	sampler = 	Sampler('single', 180, .5, (30,35), (49,50), fld=fld)
	env 	= 	Market(sampler, window_state, open_cost, time_difference=time_difference, wavelet_channels=wavelet_channels)
	agent, print_t	= 	get_agent(model_type, env, batch_size=batch_size,\
									 buffer_min_size=buffer_min_size, buffer_size=buffer_size,\
									outputdir='/home/younesz/Desktop/SUM')

	# Set save name
	fld_save = os.path.join(rootStore, 'results', sampler.title, model_type,
							str((env.window_state, sampler.window_episode, batch_size, learning_rate,
								 discount_factor, exploration_decay, env.open_cost)))

	#model.model.summary()
	#return

	visualizer	=	Visualizer(env.action_labels)


	
	print('='*20)
	print(fld_save)
	print('='*20)

	simulator = Simulator(agent, env, visualizer=visualizer, fld_save=fld_save)
	simulator.train(n_episode_training, save_per_episode=1, exploration_decay=exploration_decay, 
		exploration_min=exploration_min, print_t=print_t, exploration_init=exploration_init)
	#agent.model = load_model(os.path.join(fld_save,'model'), learning_rate)

	#print('='*20+'\nin-sample testing\n'+'='*20)
	simulator.test(n_episode_testing, save_per_episode=1, subfld='in-sample testing')

	"""
	fld = os.path.join('data',db_type,db+'B')
	sampler = SinSampler('load',fld=fld)
	simulator.env.sampler = sampler
	simulator.test(n_episode_testing, save_per_episode=1, subfld='out-of-sample testing')
	"""
	

if __name__ == '__main__':
	main()
