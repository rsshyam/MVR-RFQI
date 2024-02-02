import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math

class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0, seed=None, tight_start=False, medium_start=False,f_dynamics=None):

        # Set gym env seed
        assert not (tight_start and medium_start)
        self.seed(seed)
        self.f_dynamics=f_dynamics
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None
        self.horizon = 200
        self.periodic_dimensions = [0]
        self.tight_start = tight_start
        self.medium_start = medium_start

        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        if self.f_dynamics:
            #print('using learnt model')
            u = np.clip(u, -self.max_torque, self.max_torque)[0]
            norm_u=u/self.max_torque #normalize
            #action multiplied by force_mag is right or not for f_dynamics---It is not as GP model takes in action between -1 and 1

            #the learnt f_dynamics takes in normalized state-action and outputs normalized change in state. This corresponds to normalized environment and also it learns change in state and not next state
            #1. normalize current state and action and send as input to f-dynamics
            #2. get normalized change in state
            #3. unnormalize change in state
            #4. add unnormalized change in state to current state to obtain next state

            #if type(action)=='list':
                #self.state = self.f_dynamics(np.concatenate([state, action]))
            #    delta_s = self.f_dynamics(np.concatenate([state, action]))
            #else:
            #state_action=state.append(action) 
            #self.state = self.f_dynamics(np.reshape(np.append(state,action),(1,-1)))
            #print(action)
            #action=action/self.force_mag ##makes it between -1 and 1
            #print(action)
            #normalize state-action
            state=self.state
            norm_state=self.normalize_obs(state)
            #action to be normalized
            #print(state,norm_state)
            norm_delta_s = self.f_dynamics(np.reshape(np.append(norm_state,norm_u),(1,-1)))
            #print(norm_delta_s,'norm_delta_s')
            norm_next_state=norm_state+norm_delta_s
            #print(norm_next_state,'norm_next_state')
            #unnorm_delta_s=norm_delta_s*(self.observation_space.high-self.observation_space.low)*(1/2)
            #print(np.squeeze(np.array(state) + delta_s))
            unnorm_next_state=self.unnormalize_obs(norm_next_state)
            #print(unnorm_next_state,'unnorm_next_state')

            unnorm_theta, theta_dot = np.squeeze(unnorm_next_state)
            theta = angle_normalize(unnorm_theta)

            theta_dot = np.clip(theta_dot, -self.max_speed, self.max_speed)
            

            costs = angle_normalize(theta) ** 2 + .1 * theta_dot ** 2 + .001 * (u ** 2)
            delta_s = norm_delta_s#np.array([unnorm_newth - th, newthdot - thdot])

            self.state = np.array([theta,theta_dot])
            done=False

            return self._get_obs(), -costs, done, {'delta_obs': delta_s}



        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        unnorm_newth = th + newthdot * dt
        newth = angle_normalize(unnorm_newth)
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        costs = angle_normalize(newth) ** 2 + .1 * newthdot ** 2 + .001 * (u ** 2)
        delta_s = np.array([unnorm_newth - th, newthdot - thdot])

        self.state = np.array([newth, newthdot])
        done = False
        return self._get_obs(), -costs, done, {'delta_obs': delta_s}

    def reset(self, obs=None):
        high = np.array([np.pi, 1])
        if obs is None:
            if self.tight_start:
                self.state = self.np_random.uniform(low=[-0.35, -0.9], high=[-0.05, -0.6])
            elif self.medium_start:
                self.state = self.np_random.uniform(low=[-3, -1], high=[-1, 1])
            else:
                self.state = self.np_random.uniform(low=-high, high=high)
        else:
            self.state = obs
        self.last_u = None
        if self.f_dynamics:
            return self._get_obs()
        return self.state

    def _get_obs_f(self):
        theta, thetadot = self.state
        s=math.sin(theta)
        c=math.cos(theta)
        f_state=np.array([s,c,thetadot])
        return f_state
    
    def _get_obs(self):
        theta, thetadot = self.state
        return self.state

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    
    def normalize_obs(self, obs):
        space_size = self.observation_space.high - self.observation_space.low
        if obs.ndim == 1:
            low = self.observation_space.low
            size = space_size
        else:
            low = self.observation_space.low[None, :]
            size = space_size[None, :]
        pos_obs = obs - low
        norm_obs = (pos_obs / size * 2) - 1
        return norm_obs
    
    def unnormalize_obs(self, obs):
        space_size = self.observation_space.high - self.observation_space.low
        if obs.ndim == 1:
            low = self.observation_space.low
            size = space_size
        else:
            low = self.observation_space.low[None, :]
            size = space_size[None, :]
        obs01 = (obs + 1) / 2
        obs_ranged = obs01 * size
        unnorm_obs = obs_ranged + low
        return unnorm_obs
    

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


def pendulum_reward(x, next_obs):
    th = x[..., 0]
    thdot = x[..., 1]
    u = x[..., 2]
    costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
    return -costs
