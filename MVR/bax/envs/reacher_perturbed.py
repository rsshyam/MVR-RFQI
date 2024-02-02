import os
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from typing import Optional, List, Tuple

class BACReacherPerturbedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, tight=False, f_dynamics=None):
        self.f_dynamics=None
        utils.EzPickle.__init__(self)
        self.horizon = 50
        self.periodic_dimensions = [0, 1]
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/reacher.xml" % dir_path, 2)
        low = np.array([-np.pi, -np.pi, -0.3, -0.3, -50, -40, -0.5, -0.5])
        high = np.array([np.pi, np.pi, 0.3, 0.3, 50, 40, 0.5, 0.5])
        self.observation_space = spaces.Box(low=low, high=high)
        self.tight = tight
        self.f_dynamics=f_dynamics


        #params
        self.gravity = -9.81
        self.x_joint_damping = 1.0
        self.y_joint_damping = 1.0
        self.actuator_ctrlrange = (-1.0,1.0)
        self.actuator_ctrllimited = int(1)

    def step(self, a):
        if self.f_dynamics:
            #if not self.f_dynamics_state:
            #    self.f_dynamics_state=self._get_obs()

            #needs to be rewritten using reacher_reward
            #need to define new state just enough to be given to f_dynamics

            old_obs=self.f_dynamics_state#load current state
            #print(old_obs,'old_obs')
            norm_state=self.normalize_obs(old_obs)#normalize state
            #print(norm_state,'norm_state')
            norm_a=self.normalize_action(a)#normalize action
            #print(norm_a,'norm_a')
            norm_delta_s = self.f_dynamics(np.reshape(np.append(norm_state,norm_a),(1,-1)))#get change in state
            #print(norm_delta_s,'norm_delta_s')
            norm_next_state=norm_state+norm_delta_s#calculate normalized next stat
            #print(norm_next_state,'norm_next_state')
            unnorm_next_state=self.unnormalize_obs(norm_next_state)#unnormalize next state
            #print(unnorm_next_state,'unnorm_next_state')
            x=np.reshape(np.append(old_obs,a),(1,-1))#combine unnormalized previous state and action
            
            
            next_obs=np.squeeze(unnorm_next_state)#squeeze to correct from 2d list to 1d list

            #calculate reward
            action_dim = 2
            start_obs = x[..., :-action_dim]
            vec = next_obs[..., -2:]
            action = x[..., -action_dim:]
            reward_dist = -np.linalg.norm(vec, axis=-1)
            reward_ctrl = -np.square(action).sum(axis=-1)
            reward = reward_dist + reward_ctrl

            
            delta_obs=next_obs-old_obs
            done=False
            ob=next_obs
            ob[0]=angle_normalize(next_obs[0])
            #print(ob,'ob')
            self.f_dynamics_state=ob#update current state

            return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, delta_obs=delta_obs)



        old_obs = self._get_obs()
        self.do_simulation(a, self.frame_skip)
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        vec = vec[:2]
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl
        ob, unnorm_obs = self._get_obs(return_unnorm_obs=True)
        delta_obs = unnorm_obs - old_obs
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl, delta_obs=delta_obs)

    

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
        use_xml: bool = False,
        gravity: float = -9.81,
        actuator_ctrlrange: Tuple[float, float] = (-1.0, 1.0),
        joint_stiffness_x: float = 0.0,
        joint_stiffness_y: float = 0.0,
        springref: float = 0.0,
        joint_damping_p: float = 0.0,
        joint_frictionloss: float = 0.0
    ):
        if seed is not None:
            seed=self.seed(seed=seed)
        ob = super().reset()
        full_obs = np.concatenate([ob[:-2], np.zeros(2)])
        qpos = full_obs[:len(self.init_qpos)]
        qvel = full_obs[len(self.init_qpos):]
        self.set_state(qpos, qvel)
        check_obs = self._get_obs()

        # grab model
        model = self.sim.model
        # perturb gravity in z (3rd) dimension*
        model.opt.gravity[2] = gravity
        # perturb x joint*
        model.jnt_stiffness[0] = joint_stiffness_x
        model.qpos_spring[0] = springref
        # perturb y joint*
        model.jnt_stiffness[1] = joint_stiffness_y
        model.qpos_spring[1] = springref
        # perturb actuator (controller) control range*
        model.actuator_ctrllimited[0] = self.actuator_ctrllimited
        model.actuator_ctrlrange[0] = [actuator_ctrlrange[0],
                                        actuator_ctrlrange[1]]
        model.actuator_ctrllimited[1] = self.actuator_ctrllimited
        model.actuator_ctrlrange[1] = [actuator_ctrlrange[0],
                                        actuator_ctrlrange[1]]
        # perturb joint damping in percentage
        model.dof_damping[0] = self.x_joint_damping * (1 + joint_damping_p) 
        model.dof_damping[1] = self.y_joint_damping * (1 + joint_damping_p) 
        # perturb joint frictionloss
        model.dof_frictionloss[0] = joint_frictionloss
        model.dof_frictionloss[1] = joint_frictionloss
        return check_obs
    

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos_max = 0.1
        qpos_min = 0.07 if self.tight else -0.1
        qpos = (
            self.np_random.uniform(low=qpos_min, high=qpos_max, size=self.model.nq)
            + self.init_qpos
        )
        if self.tight:
            goal_low = -0.05
            goal_high = -0.03
        else:
            goal_low = -0.2
            goal_high = 0.2
        while True:
            self.goal = self.np_random.uniform(low=goal_low, high=goal_high, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self, return_unnorm_obs=False):
        theta = self.sim.data.qpos.flat[:2]
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        vec = vec[:2]
        norm_theta = angle_normalize(theta)
        obs = np.concatenate(
            [
                # np.cos(theta),
                # np.sin(theta),
                norm_theta,
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                vec,
            ]
        )
        if return_unnorm_obs:
            unnorm_obs = np.concatenate(
                [
                    # np.cos(theta),
                    # np.sin(theta),
                    theta,
                    self.sim.data.qpos.flat[2:],
                    self.sim.data.qvel.flat[:2],
                    vec,
                ]
            )
            return obs, unnorm_obs
        else:
            return obs
    
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
    
    def normalize_action(self, action):
        space_size = self.action_space.high - self.action_space.low
        if action.ndim == 1:
            low = self.action_space.low
            size = space_size
        else:
            low = self.action_space.low[None, :]
            size = space_size[None, :]
        pos_action = action - low
        norm_action = (pos_action / size * 2) - 1
        return norm_action
    
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

def reacher_reward(x, next_obs):
    action_dim = 2
    start_obs = x[..., :-action_dim]
    vec = next_obs[..., -2:]
    action = x[..., -action_dim:]
    reward_dist = -np.linalg.norm(vec, axis=-1)
    reward_ctrl = -np.square(action).sum(axis=-1)
    reward = reward_dist + reward_ctrl
    return reward


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


if __name__ == "__main__":
    # we have an increased ATOL because the COM part of the state is the solution
    # to some kind of FK problem and can have numerical error
    env = BACReacherEnv()
    print(f"{env.observation_space=}, {env.action_space=}")
    og_obs = env.reset()
    obs = og_obs
    done = False
    for _ in range(env.horizon):
        action = env.action_space.sample()
        next_obs, rew, done, info = env.step(action)
        x = np.concatenate([obs, action])
        other_rew = reacher_reward(x, next_obs)
        assert np.allclose(rew, other_rew, atol=1e-2), f"{rew=}, {other_rew=}"
        obs = next_obs
        new_obs = env.reset(obs)
        assert np.allclose(new_obs, obs, atol=1e-2), f"{new_obs=}, {obs=}"
    # test reset to point
    env.reset(og_obs)


