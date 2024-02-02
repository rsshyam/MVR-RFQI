import numpy as np
from stable_baselines3 import PPO, SAC, DQN, TD3
import gym
from gym import spaces
from data_container import DATA
import os
import argparse
import torch

from argparse import Namespace
import logging
import numpy as np
import gym
from tqdm import trange
from functools import partial
import tensorflow as tf
import hydra
import random
from matplotlib import pyplot as plt

from bax.models.gpfs_gp import BatchMultiGpfsGp
from bax.models.gpflow_gp import get_gpflow_hypers_from_data
from bax.acq.acquisition import MultiBaxAcqFunction, MCAcqFunction, UncertaintySamplingAcqFunction
from bax.acq.acqoptimize import AcqOptimizer
from bax.alg.mpc import MPC
from bax import envs
from bax.envs.wrappers import NormalizedEnv, make_normalized_reward_function, make_update_obs_fn
from bax.envs.wrappers import make_normalized_plot_fn
from bax.util.misc_util import Dumper, make_postmean_fn
from bax.util.control_util import get_f_batch_mpc, get_f_batch_mpc_reward, compute_return, evaluate_policy
from bax.util.control_util import rollout_mse, mse
from bax.util.domain_util import unif_random_sample_domain
from bax.util.timing import Timer
from bax.viz import plotters, make_plot_obs
import neatplot

import torch


from rlkit.policies.base import Policy
from rlkit.torch.sac.policies import MakeDeterministic,TanhGaussianPolicy

def load_rlkit_policy(path: str, policy_name: str, state_dim : int, action_dim : int, make_deterministic: bool = True) -> Policy:
    """Load in an rlkit policy
    Args:
        path to weights.pt: weight of the networks
        make_deterministic: whether to make the policy to return deterministic

    Returns:
        The loaded policy
    """
    M=128
    policy = TanhGaussianPolicy(obs_dim=state_dim,action_dim=action_dim,hidden_sizes=[M, M],)
    policy.load_state_dict(torch.load(os.path.join(path,policy_name),'cpu'))
    policy=policy.cpu()
    if make_deterministic:
        policy = MakeDeterministic(policy)
    return policy

def get_action_type(action_space):
    """
    Method to get the action type to choose prob. dist. 
    to sample actions from NN logits output.
    """
    if isinstance(action_space, spaces.Box):
        shape = action_space.shape
        assert len(shape) == 1
        if shape[0] == 1:
            return 'continuous'
        else:
            return 'multi_continuous'
    elif isinstance(action_space, spaces.Discrete):
        return 'discrete'
    elif isinstance(action_space, spaces.MultiDiscrete):
        return 'multi_discrete'
    elif isinstance(action_space, spaces.MultiBinary):
        return 'multi_binary'
    else:
        raise NotImplementedError
        
def generate_dataset(env_name, gendata_pol, epsilon, state_dim, action_dim,
                     args, postmean_fn, nsamples, normalize_env=False, buffer_size=int(1e6), verbose=False,mpc_func=None):
    # determine trained policy save path and where to save dataset
   
    if args.mixed == 'True':
        dataset_name = f'./offline_data/{env_name}_{gendata_pol}_mixed_e{epsilon}_nsamples_{nsamples}'
        policy_path = f'models/{gendata_pol}_mixed_{env_name}'
    else:
        dataset_name = f'./offline_data/{env_name}_{gendata_pol}_e{epsilon}_nsamples_{nsamples}'
        if gendata_pol!='mpc' and gendata_pol!='rlkit_sac':
            policy_path = f'{gendata_pol}_{env_name}_{nsamples}'

    
    if epsilon < 1:
        if gendata_pol == 'ppo':
            policy = PPO.load(policy_path, device=args.device)
        elif gendata_pol == 'sac':
            policy = SAC.load(policy_path, device=args.device)
        elif gendata_pol == 'dqn':
            policy = DQN.load(policy_path, device=args.device)
        elif gendata_pol == 'td3':
            policy = TD3.load(policy_path, device=args.device)
        elif gendata_pol == 'mpc':
            policy = mpc_func
        elif gendata_pol == 'torch_sac':
            M=128
            sac_policy = TanhGaussianPolicy(obs_dim=state_dim,action_dim=action_dim,hidden_sizes=[M, M],)
            #sac_policy.load_state_dict(torch.load('policy.pt'),'cuda')
            sac_policy.load_state_dict(torch.load('best_eval_policy.pt'),'cuda')
            policy=sac_policy.cuda()
            #policy_name = f'{gendata_pol}_{env_name}_{nsamples}'
            #policy_path = f'rlkit_policies'
            #policy = load_rlkit_policy(policy_path,policy_name,state_dim,action_dim)
        else:
            raise NotImplementedError
    #env_name_frpo=policy.env
    if args.mixed == 'True':
        dataset_name = f'./offline_data/{env_name}_{gendata_pol}_mixed_e{epsilon}_nsamples_{nsamples}'
        #policy_path = f'models/{gendata_pol}_mixed_{env_name}'
    else:
        dataset_name = f'./offline_data/{env_name}_{gendata_pol}_e{epsilon}_nsamples_{nsamples}'
    #print(policy.observation_space.shape)
    #if policy.observation_space.shape[0]!=state_dim:
    #    state_dim=policy.observation_space.shape[0]
    #    print(state_dim)
    # prep. environment
    env = gym.make(env_name,f_dynamics=postmean_fn)
    env_hor=env.horizon
    reward_function = envs.reward_functions[env_name]
    if normalize_env:
        env = NormalizedEnv(env)
        #if reward_function is not None:
        #    reward_function = make_normalized_reward_function(plan_env, reward_function)
    env_action_type = get_action_type(env.action_space)
    #print(buffer_size,'buffer_size')
    data = DATA(state_dim, action_dim, 'cpu', buffer_size)
    states = []
    actions = []
    next_states = []
    # next_states_lrnt=[]
    rewards = []
    not_dones = []
    
    # set path
    if args.mixed == 'True':
        dataset_name = f'./offline_data/{env_name}_{gendata_pol}_mixed_e{epsilon}_nsamples_{nsamples}'
    else:
        dataset_name = f'./offline_data/{env_name}_{gendata_pol}_e{epsilon}_nsamples_{nsamples}'
    
    # generate dateset
    count = 0
    #print(env.observation_space)
    while count < buffer_size:
        state, done = env.reset(), False
        count_done=count
        #print('new')
        if verbose:
            print(f'buffer size={buffer_size}======current count={count}')
        while not done:
            if epsilon < 1:
                if np.random.binomial(n=1, p=epsilon):
                    action = env.action_space.sample()
                else: # else we select expert action
                    if gendata_pol=='mpc':
                        action=policy(state)
                    elif gendata_pol=='torch_sac':
                        #print('hello')
                        action=policy.get_action(state)
                        #print('bye')
                        if isinstance(action,tuple):#PPO outputs a tuple with action as first element
                            action=action[0]
                            #print(action,'from policy')
                    else:
                        action, _ = policy.predict(state)
                    if 'FrozenLake' in env_name:
                        action = int(action)
            else:
                action = env.action_space.sample()
                #print(action,'random')
            #collect reward from environment
            next_state, reward, done, _ = env.step(action)
            #print(next_state)
           

            #refer process_prevoutput for delta and change in state

            #collect next state from postmean_fn
            #query=np.concatenate([state,action],axis=1)
            #next_state_lrnt=postmean_fn(query)
            
            #print(state,action)
            #print(env.observation_space)
            states.append(env.unnormalize_obs(state))
            # determine the correct data structure for action
            if env_action_type == 'continuous' or env_action_type == 'discrete':
                action = np.array([action])
            elif env_action_type == 'multi_continuous' or env_action_type == 'multi_discrete' or env_action_type == 'multi_binary':
                action = np.array(action)
            else:
                raise NotImplementedError
                
            if np.random.binomial(n=1, p=0.001):
                print('==================================================')
                print('--------random printing offline data point--------')
                print(f'state: {state}')
                print(f'action: {action}')
                print(f'next_state: {next_state}')
                print(f'unnormalized_state: {env.unnormalize_obs(state)}')
                print(f'action: {env.unnormalize_action(action)}')
                print(f'next_state: {env.unnormalize_obs(next_state)}')
                #print(f'next_state: {next_state_lrnt}')
                print(f'not_done: {1.0 - done}')
                print(f'reward: {reward}')
            actions.append(env.unnormalize_action(action))
            next_states.append(env.unnormalize_obs(next_state))
            #next_states_lrnt.append(next_states_lrnt)
            not_dones.append(np.array([1.0 - done]))
            rewards.append(np.array([reward]))
        
            # check buffer size
            count += 1
            if count >= buffer_size:
                break
            elif count >=count_done+env_hor:
                break            
            else:    # state transition
                state = next_state
            
        
    data.state = np.array(states)
    data.state = np.resize(data.state, (buffer_size, state_dim))
    data.action = np.array(actions)
    data.action = np.resize(data.action, (buffer_size, action_dim))
    data.next_state = np.array(next_states)
    data.next_state = np.resize(data.next_state, (buffer_size, state_dim))
    #data.next_state_lrnt = np.array(next_states_lrnt)
    #data.next_state_lrnt = np.resize(data.next_state_lrnt, (buffer_size, state_dim))
    data.reward = np.array(rewards)
    data.not_done = np.array(not_dones)
    data.size = buffer_size
    data.ptr = buffer_size
    data.save(dataset_name)

     
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # OpenAI gym environment name (need to be consistent with the dataset name)
    parser.add_argument("--env", default='CartPole-v1')
    # e-mix (prob. to mix random actions)
    parser.add_argument("--eps", default=0.5, type=float)
    parser.add_argument("--buffer_size", default=1e6, type=float)
    parser.add_argument("--verbose", default='False', type=str)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--gendata_pol", default='ppo', type=str)
    # if gendata_pol is trained with mixed traj.
    parser.add_argument("--mixed", default='False', type=str)
    args = parser.parse_args()
    
    if args.verbose == 'False':
        verbose = False
    else:
        verbose = True
        
    # determine dimensions
    env = gym.make(args.env)
    env_action_type = get_action_type(env.action_space)
    if env_action_type == 'continuous':
        action_dim = 1
        max_action = env.action_space.high
        min_action = env.action_space.low
    elif env_action_type == 'discrete':
        action_dim = 1
        max_action = env.action_space.n - 1
        min_action = 0
    elif env_action_type == 'multi_continuous':
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high
        min_action = env.action_space.low
    elif env_action_type == 'multi_discrete':
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.nvec.max()
        min_action = env.action_space.nvec.min()
    elif env_action_type == 'multi_binary':
        action_dim = env.action_space.n
        max_action = 1
        min_action = 0
    else:
        raise NotImplementedError
    
    if isinstance(env.observation_space, spaces.Discrete):
        state_dim = 1
    else:
        state_dim = env.observation_space.shape[0]
    
    # client input sanity check
    if args.device not in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'auto']:
        raise NotImplementedError
        
    # check path
    if not os.path.exists("./offline_data"):
        os.makedirs("./offline_data")
        
    # check mixed option
    if args.mixed == 'True' or args.mixed == 'False':
        pass
    else:
        raise NotImplementedError
        
    generate_dataset(args.env, args.gendata_pol, args.eps, state_dim, action_dim,
                     args, int(args.buffer_size), verbose)
