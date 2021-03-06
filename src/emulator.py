import matplotlib.pyplot as plt

from src.lib import *
from itertools import compress
from scripts.dwt_no_edge_effects import *

# by Xiang Gao, 2018



def find_ideal(p, just_once):
	if not just_once:
		diff = np.array(p[1:]) - np.array(p[:-1])
		return sum(np.maximum(np.zeros(diff.shape), diff))
	else:
		best = 0.
		i0_best = None
		for i in range(len(p)-1):
			best = max(best, max(p[i+1:]) - p[i])

		return best


class Market:
	"""
	state 			MA of prices, normalized using values at t
					ndarray of shape (window_state, n_instruments * n_MA), i.e., 2D
					which is self.state_shape

	action 			three action
					0:	go short (sell/wait)
					1:	go long (buy/keep)
	"""
	
	def reset(self, rand_price=True):
		self.empty = 1.
		if rand_price:
			prices, self.title = self.sampler.sample()
			price = np.reshape(prices[:,0], prices.shape[0])

			self.prices = prices.copy()
			self.price = price/price[0]*100
			self.t_max = len(self.price) - 1

		self.max_profit = find_ideal(self.price[self.t0:], False)
		self.t = self.t0
		return self.get_state(), self.get_valid_actions()


	def get_state(self, t=None):
		if t is None:
			t = self.t
		state = self.prices[t - self.window_state + 1 - self.time_difference: t + 1, :].copy()
		# --- Time-difference
		if self.time_difference:
			state=	state[1:] - state[:-1]
		# --- Wavelet transform
		if self.wavelet_channels>0:
			state=	dwt_no_edge_effects(state, nlevels=self.wavelet_channels)
		for i in range(self.sampler.n_var):
			norm = np.mean(state[:,i])
			stdv = np.std(state[:,i])
			state[:,i] = (( state[:,i]-norm)/stdv )#*100
		return state.T


	def get_valid_actions(self):
		valid_actions	=	[self.empty-1, self.empty]	# negative: sell %age of BTC, positive: buy %age of BTC
		if not self.action_conversion is None:
			ls_actions 	=	list( range(self.n_action) )
			filter1		=	self.action_conversion >= valid_actions[0]
			filter2 	= 	self.action_conversion <= valid_actions[1]
			mask 		=	[all(x) for x in zip(filter1, filter2)]
			valid_actions 	=	list( compress(ls_actions, mask ) )
		return valid_actions


	def get_noncash_reward(self, t=None, empty=None):
		if t is None:
			t = self.t
		if empty is None:
			empty = self.empty
		reward = self.direction * (self.price[t+1] - self.price[t])
		if empty:
			reward -= self.open_cost*self.price[t]/100
		if reward < 0:
			reward *= (1. + self.risk_averse)
		return reward


	def step(self, action):

		# Convert the action
		if not self.action_conversion is None:
			action	=	self.action_conversion[action]
		elif self.n_action==1:
			# Clip to valid range
			action 	=	np.minimum( np.maximum(action, self.empty-1), self.empty )
		done	= 	False
		# The reward now has two components: asset-moving, asset-keeping
		reward 	=	0.
		# Moved asset
		if action>0:
			# Bought BTC
			reward 	+=	self.get_noncash_reward(empty=True) * action
		# Kept asset
		#reward 		+=	self.get_noncash_reward(empty=False) * (1 - self.empty + min(0,action))
		self.empty 	-= 	action  # update status

		self.t += 1
		return self.get_state(), reward, self.t == self.t_max, self.get_valid_actions()


	def __init__(self, sampler, window_state, open_cost, action_labels=['continuous'], action_range=None,
				 wavelet_channels=0, direction=1., risk_averse=0., time_difference=True):

		self.sampler 		= 	sampler
		self.window_state 	= 	window_state
		self.open_cost 		= 	open_cost
		self.direction 		= 	direction
		self.risk_averse 	= 	risk_averse
		self.time_difference=	True
		self.wavelet_channels=	wavelet_channels
		self.state_shape 	= 	(self.sampler.n_var, window_state)
		self.action_labels 	= 	action_labels	#['short all', 'short half', 'hold', 'long half', 'long all']	#['empty','open','keep']
		self.n_action 		= 	len(self.action_labels)
		self.t0 			= 	window_state - 1 + self.time_difference
		self.train_window 	=	self.sampler.window_episode - self.window_state - self.time_difference

		# Action conversion
		self.action_conversion 		=	None
		if not action_range is None and self.n_action>1:
			self.action_conversion 	=	np.linspace(action_range[0], action_range[1], self.n_action)


if __name__ == '__main__':
	test_env()
