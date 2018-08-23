import numpy as np
import matplotlib.pyplot as plt


class Sine():

    def __init__(self, action_size=2):
        self.observation_space  =   2
        self.action_space       =   action_size
        self.action_range       =   [-1, 1]
        self.allowed_actions    =   np.linspace(self.action_range[0], self.action_range[1], action_size)
        self.spec               =   {'timestep_limit':100, 'period':20, 'phase':np.random.random()*2*np.pi}
        self.render_vars        =   None
        self.reset()

    def reset(self):
        self.x, self.y  =   0, np.sin(-self.spec['phase'])
        self.nsteps     =   0
        return np.reshape([np.sin( (self.x-1)*2*np.pi/self.spec['period'] - self.spec['phase']), self.y], [1,-1])

    def step(self, action):
        self.x  +=  1
        self.y  +=  self.allowed_actions[action]
        reward  =   - np.sqrt( (self.y - np.sin(self.x*2*np.pi/self.spec['period'] - self.spec['phase']))**2 )
        return np.reshape([np.sin( (self.x-1)*2*np.pi/self.spec['period'] - self.spec['phase']), self.y], [1,-1]), reward.flatten()[0], self.x==self.spec['timestep_limit'], ''

    def run_demo(self):
        n_steps     =   0
        s   =   self.reset()
        while n_steps<self.spec['timestep_limit']:
            # Agent act
            action          =   np.random.randn()
            s_, r, done, _  =   self.step(action)
            # Print
            self.render(s_, r)
            n_steps +=  1

    def render(self, s, a, r, pause_time=0.3):
        tm = np.arange(self.spec['timestep_limit'])
        if self.render_vars is None:
            # Draw the sine function
            self.render_vars    =   {'F'    :   plt.figure()}
            self.render_vars['Ax']  =   self.render_vars['F'].add_subplot(111)
            self.render_vars['Ax'].plot(tm, np.sin(tm * 2 * np.pi / self.spec['period'] - self.spec['phase']))
            self.render_vars['pl']  =   self.render_vars['Ax'].plot(self.x, self.y, color='green', marker='o')
            self.render_vars['tt']  =   self.render_vars['Ax'].text(1, 1, "game started")
            plt.pause(pause_time)

        # Hide previous state and reward
        self.render_vars['tt'].set_visible(False)
        [x.set_visible(False) for x in self.render_vars['pl']]

        # Print current state and reward
        self.render_vars['pl']  =   self.render_vars['Ax'].plot( np.reshape(np.add(self.x, [-1, 0]), [1, -1]), s, color='green', marker='o')
        self.render_vars['tt']  =   self.render_vars['Ax'].text(1, 1, "action=%.1f, reward=%.2f" %(a, r))
        plt.pause(pause_time)

    def kill(self):
        plt.close(self.render_vars['F'])
        self.render_vars    =   None


    def seed(self, new_seed):
        np.random.seed(new_seed)


# Launcher
if __name__=='__main__':
    env     =   Sine()
    env.run_demo()