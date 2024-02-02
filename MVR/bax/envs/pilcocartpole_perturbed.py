"""
Cart pole swing-up: Identical version to PILCO V0.9
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class CartPolePerturbedEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    OBSERVATION_DIM = 4
    POLE_LENGTH = 0.6
    #
    def __init__(self, use_trig=False, f_dynamics=None):
        self.f_dynamics = f_dynamics
        self.use_trig = use_trig
        self.g = 9.82  # gravity
        self.m_c = 0.5  # cart mass
        self.m_p = 0.5  # pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.l = CartPolePerturbedEnv.POLE_LENGTH # pole's length
        self.m_p_l = (self.m_p*self.l)
        self.force_mag = 10.0
        self.dt = 0.1  # seconds between state updates
        self.b = 0.1  # friction coefficient
        self.horizon = 25
        self.periodic_dimensions = [2]

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array([
            10.,
            10.,
            3.14159,
            25.])

        self.action_space = spaces.Box(-1, 1, shape=(1,))
        self.observation_space = spaces.Box(-high, high)
        if self.use_trig:
            high = np.array([10., 10., 1., 1., 25.])
            self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Valid action
        action = np.clip(action, -1, 1)[0] * self.force_mag
        #print(action,'action')
        state = self.state
        x, x_dot, theta, theta_dot = state
        #print(state,'step begin')
        if self.f_dynamics:
            if type(action)=='list':
                #self.state = self.f_dynamics(np.concatenate([state, action]))
                delta_s = self.f_dynamics(np.concatenate([state, action]))
            else:
                #state_action=state.append(action) 
                #self.state = self.f_dynamics(np.reshape(np.append(state,action),(1,-1)))
                delta_s = self.f_dynamics(np.reshape(np.append(state,action),(1,-1)))
                #print(np.squeeze(np.array(state) + delta_s))
                x, x_dot, unnorm_theta, theta_dot = np.squeeze(np.array(state) + delta_s)
                theta = angle_normalize(unnorm_theta)
                self.state = (x,x_dot,theta,theta_dot)
        else:
            s = math.sin(theta)
            c = math.cos(theta)
            #print(s,c)
            xdot_update = (-2*self.m_p_l*(theta_dot**2)*s + 3*self.m_p*self.g*s*c + 4*action - 4*self.b*x_dot)/(4*self.total_m - 3*self.m_p*c**2)
            #print(xdot_update,'xdot_update')
            thetadot_update = (-3*self.m_p_l*(theta_dot**2)*s*c + 6*self.total_m*self.g*s + 6*(action - self.b*x_dot)*c)/(4*self.l*self.total_m - 3*self.m_p_l*c**2)
            #print(thetadot_update,'thetadot_update')
            x = x + x_dot*self.dt
            #print(x,'x')
            unnorm_theta = theta + theta_dot*self.dt
            #print(unnorm_theta,'unnorm_theta')
            theta = angle_normalize(unnorm_theta)
            #print(theta,'theta')
            x_dot = x_dot + xdot_update*self.dt
            #print(x_dot,'x_dot')
            theta_dot = theta_dot + thetadot_update*self.dt
            #print('theta_dot',theta_dot)
            delta_s = np.array([x, x_dot, unnorm_theta, theta_dot]) - np.array(state)
            #print(delta_s,'delta_s')
            self.state = (x,x_dot,theta,theta_dot)
            #print(self.state,'end')
        
        # use postmean here instead of 69-80
        # compute costs - saturation cost
        goal = np.array([0.0, self.l])
        pole_x = self.l*np.sin(theta)
        pole_y = self.l*np.cos(theta)
        position = np.array([self.state[0] + pole_x, pole_y])
        squared_distance = np.sum((position - goal)**2)
        squared_sigma = 0.25**2
        costs = 1 - np.exp(-0.5*squared_distance/squared_sigma)
        done=False
        #done = bool(
        #    x < -self.x_threshold
        #    or x > self.x_threshold
        #    or theta < -self.theta_threshold_radians
        #    or theta > self.theta_threshold_radians
        #)
        #print(done)
        #if not done:
        #    reward = 1.0
        #elif self.steps_beyond_done is None:
        #    # Pole just fell!
        #    self.steps_beyond_done = 0
        #    reward = 1.0
        #    costs=1000#arbitrarily large value
        #else:
            #print(done)
        #    if self.steps_beyond_done == 0:
        #        logger.warn(
        #            "You are calling 'step()' even though this "
        #            "environment has already returned done = True. You "
        #            "should always call 'reset()' once you receive 'done = "
        #            "True' -- any further steps are undefined behavior."
        #        )
        #    self.steps_beyond_done += 1
        #    costs=1000#arbitrarily large value








        return self.get_obs(), -costs, done, {'delta_obs': delta_s}

    def get_obs(self):
        if self.use_trig:
            return np.array([self.state[0], self.state[1], np.sin(self.state[2]), np.cos(self.state[2]), self.state[3]])
        else:
            return np.array(self.state)


    def reset(self, obs=None, seed=None, length=0.5, gravity=9.8, force_mag=10.0, 
              init_angle_mag=0.05, init_vel_mag=0.05):
        # set seed if any
        if seed is not None:
            #super().reset(seed=seed)
            seed=self._seed(seed=seed)
        # perturbing the environment
        self.g = gravity
        self.force_mag = force_mag
        self.l = length
        self.m_p_l = self.m_p * self.l
        init_angle = np.random.uniform(-init_angle_mag, init_angle_mag)
        init_vel = np.random.uniform(-init_vel_mag, init_vel_mag)
        # build state
        # self.state[2] is initial angle
        # self.state[3] is initial velocity


        #how to initialize self.state normal or uniform

        if obs is None:
            self.state = self.np_random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.02, 0.02, 0.02, 0.02]))
            self.state[2] = init_angle
            self.state[3] = init_vel        
        else:
            assert not self.use_trig, f"can't use trig if you are going to have generative access"
            self.state = obs
        #self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)) # changed from 4 to 2
        self.steps_beyond_done = None
        #print(self.state,'before')
        return np.array(self.state, dtype=np.float32)



    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = 5  # max visible position of cart
        scale = screen_width/world_width
        carty = 200 # TOP OF CART
        polewidth = 6.0
        polelen = scale*self.l  # 0.6 or self.l
        cartwidth = 40.0
        cartheight = 20.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2

            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(1, 0, 0)
            self.viewer.add_geom(cart)

            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(0, 0, 1)
            self.poletrans = rendering.Transform(translation=(0, 0))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.1, 1, 1)
            self.viewer.add_geom(self.axle)

            # Make another circle on the top of the pole
            self.pole_bob = rendering.make_circle(polewidth/2)
            self.pole_bob_trans = rendering.Transform()
            self.pole_bob.add_attr(self.pole_bob_trans)
            self.pole_bob.add_attr(self.poletrans)
            self.pole_bob.add_attr(self.carttrans)
            self.pole_bob.set_color(0, 0, 0)
            self.viewer.add_geom(self.pole_bob)


            self.wheel_l = rendering.make_circle(cartheight/4)
            self.wheel_r = rendering.make_circle(cartheight/4)
            self.wheeltrans_l = rendering.Transform(translation=(-cartwidth/2, -cartheight/2))
            self.wheeltrans_r = rendering.Transform(translation=(cartwidth/2, -cartheight/2))
            self.wheel_l.add_attr(self.wheeltrans_l)
            self.wheel_l.add_attr(self.carttrans)
            self.wheel_r.add_attr(self.wheeltrans_r)
            self.wheel_r.add_attr(self.carttrans)
            self.wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
            self.wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
            self.viewer.add_geom(self.wheel_l)
            self.viewer.add_geom(self.wheel_r)



            self.track = rendering.Line((0,carty - cartheight/2 - cartheight/4), (screen_width,carty - cartheight/2 - cartheight/4))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[2])
        self.pole_bob_trans.set_translation(-self.l*np.sin(x[2]), self.l*np.cos(x[2]))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


def get_pole_pos(x):
    xpos = x[..., 0]
    theta = x[..., 2]
    pole_x = CartPolePerturbedEnv.POLE_LENGTH*np.sin(theta)
    pole_y = CartPolePerturbedEnv.POLE_LENGTH*np.cos(theta)
    position = np.array([xpos + pole_x, pole_y]).T
    return position


def pilco_cartpole_reward(x, next_obs):
    position = get_pole_pos(next_obs)
    goal = np.array([0.0, CartPolePerturbedEnv.POLE_LENGTH])
    squared_distance = np.sum((position - goal)**2, axis=-1)
    squared_sigma = 0.25**2
    costs = 1 - np.exp(-0.5*squared_distance/squared_sigma)
    return -costs


def test_cartpole():
    env = CartPolePerturbedEnv()
    n_tests = 100
    for _ in range(n_tests):
        obs = env.reset()
        action = env.action_space.sample()
        next_obs, rew, done, info = env.step(action)
        x = np.concatenate([obs, action])
        other_rew = pilco_cartpole_reward(x, next_obs)
        assert np.allclose(rew, other_rew)
        new_obs = env.reset(obs)
        assert np.allclose(new_obs, obs)
    done = False
    env.reset()
    for _ in range(env.horizon):
        action = env.action_space.sample()
        n, r, done, info = env.step(action)
        if done:
            break
    print("passed")


if __name__ == '__main__':
    test_cartpole()
