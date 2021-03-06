""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp

from external_modules.patemami_DDPG.ddpg.replay_buffer import ReplayBuffer

# ===========================
#   Actor and Critic DNNs
# ===========================
class DDPG():

    def __init__(self, state_dim, action_dim, action_bound=1, lr=[1e-4, 1e-3], tau=0.001, batch_size=64, gamma=0.99, seed=np.random.randint(1e9), buffer_size=2000, outputdir='/home/younesz/Desktop/SUM'):
        # Initialize TF session
        self.saver  =   tf.train.Saver()
        self.sess   =   tf.Session()
        np.random.seed(seed)
        tf.set_random_seed(seed)
        # Create actor
        self.Actor  =   ActorNetwork(self.sess, state_dim, action_dim, action_bound, lr[0], tau, batch_size)
        # Create critic
        self.Critic =   CriticNetwork(self.sess, state_dim, action_dim, lr[1], tau, gamma, self.Actor.get_num_trainable_vars())
        # Critic noise process
        self.Noise  =   OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
        # Initialize replay memory
        self.Buffer =   ReplayBuffer(buffer_size, seed)

        # Initialize summary operations
        self.summary_ops, self.summary_vars = build_summaries()
        self.sess.run(tf.global_variables_initializer())
        self.writer =   tf.summary.FileWriter(outputdir, self.sess.graph)


    def act(self, s, exploration=0.1, valid_actions=[-1,1]):
        return self.Actor.predict(np.reshape(s, (1, self.Actor.s_dim))) + self.Noise()

    def remember(self, s, a, r, s2, terminal, valid_actions=[-1, 1]):
        self.Buffer.add(np.reshape(s, (self.Actor.s_dim,)), np.reshape(a, (self.Actor.a_dim,)), r, np.reshape(s2, (self.Actor.s_dim,)), terminal)

    def replay(self, trace=1):
        predicted_q_value   =   0
        if self.Buffer.size() > self.Actor.batch_size:
            s_batch, a_batch, r_batch, s2_batch, t_batch    =   self.Buffer.sample_batch(self.Actor.batch_size)

            # Calculate targets
            target_q    =   self.Critic.predict_target(s2_batch, self.Actor.predict_target(s2_batch))

            y_i = []
            for k in range(self.Actor.batch_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.Critic.gamma * target_q[k])

            # Update the critic given the targets
            predicted_q_value, _    =   self.Critic.train(s_batch, a_batch, np.reshape(y_i, (self.Actor.batch_size, 1)))

            # Update the actor policy using the sampled gradient
            a_outs  =   self.Actor.predict(s_batch)
            grads   =   self.Critic.action_gradients(s_batch, a_outs)
            self.Actor.train(s_batch, grads[0])

            # Update target networks
            self.Actor.update_target_network()
            self.Critic.update_target_network()
        return np.amax(predicted_q_value)

    def update_summary(self, ep_reward, ep_ave_max_q, episode, n_steps):
        summary_str = self.sess.run(self.summary_ops, feed_dict={self.summary_vars[0]: ep_reward, self.summary_vars[1]: ep_ave_max_q/n_steps})
        self.writer.add_summary(summary_str, episode)
        self.writer.flush()
        print('| Reward: {:d} | Episode: {:d} | N_steps: {:d} | Qmax: {:.4f}'.format(int(ep_reward), episode, n_steps, (ep_ave_max_q / n_steps)))

    def save(self, path):
        return self.saver.save(self.sess, path)

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                  + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess   =   sess
        self.s_dim  =   state_dim
        self.a_dim  =   action_dim
        self.learning_rate  =   learning_rate
        self.tau    =   tau
        self.gamma  =   gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================

def train(env, args, agent):



    # Initialize target network weights
    agent.Actor.update_target_network()
    agent.Critic.update_target_network()

    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward   =   0
        ep_ave_max_q=   0

        for j in range(int(args['max_episode_len'])):

            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = agent.act(s)

            s2, r, terminal, info = env.step(a[0])

            if args['render_env']:
                env.render(s, r)

            agent.remember(s, a, r, s2, terminal)
            """agent.Buffer.add(np.reshape(s, (agent.Actor.s_dim,)), np.reshape(a, (agent.Actor.a_dim,)), r,
                              terminal, np.reshape(s2, (agent.Actor.s_dim,)))
            """

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            ep_ave_max_q    +=  agent.replay()
            """
            if agent.Buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    agent.Buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = agent.Critic.predict_target(
                    s2_batch, agent.Actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + agent.Critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = agent.Critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = agent.Actor.predict(s_batch)
                grads = agent.Critic.action_gradients(s_batch, a_outs)
                agent.Actor.train(s_batch, grads[0])

                # Update target networks
                agent.Actor.update_target_network()
                agent.Critic.update_target_network()
            """
            s = s2
            ep_reward += r

            if terminal:
                agent.update_summary(ep_reward, ep_ave_max_q, i, j)
                break

def main(args):

    with tf.Session() as sess:

        env = gym.make(args['env'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
        # Ensure action bound is symmetric
        assert (env.action_space.high == -env.action_space.low)

        """
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
        """

        Agent = DDPG(state_dim, action_dim, action_bound=action_bound, \
                     lr=[float(args['actor_lr']), float(args['critic_lr'])], \
                     tau=float(args['tau']), \
                     batch_size=int(args['minibatch_size']), \
                     gamma=float(args['gamma']), \
                     seed=int(args['random_seed']), \
                     outputdir=str(args['summary_dir']))


        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(env, args, Agent)

        if args['use_gym_monitor']:
            env.monitor.close()


def test_sine(args):
    from  generic.environments import Sine

    env = Sine()
    np.random.seed(int(args['random_seed']))
    tf.set_random_seed(int(args['random_seed']))
    env.seed(int(args['random_seed']))
    args['max_episode_len']     =   env.spec['timestep_limit']
    #env.seed(int(args['random_seed']))

    state_dim   =   env.observation_space
    action_dim  =   env.action_space
    action_bound=   1

    Agent   =   DDPG(state_dim, action_dim, action_bound=action_bound,\
                     lr         =   [float(args['actor_lr']), float(args['critic_lr'])],\
                     tau        =   float(args['tau']),\
                     batch_size =   int(args['minibatch_size']),\
                     gamma      =   float(args['gamma']),\
                     seed       =   int(args['random_seed']),\
                     outputdir  =   str(args['summary_dir']))

    train(env, args, Agent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    #main(args)
    test_sine(args)
