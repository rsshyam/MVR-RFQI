"""
Testing MultiGpfsGp and MultiBaxAcqFunction classes
"""
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
import stable_baselines3
from stable_baselines3 import PPO,SAC,TD3
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
from neptune import management
import neptune.new as neptune
from stable_baselines3.common.noise import NormalActionNoise
from gen_data_file_lrnt_mod import get_action_type
from gym import spaces
import os
from gen_data_file_lrnt_mod import generate_dataset
#from rl-baselines3-zoo import train.py
from stable_baselines3.common.logger import configure

import wandb
import imageio


#######################################
#######bac_baselines###################
#######################################
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.core import logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from bac_baselines.sac import experiment as sac_experiment
from bac_baselines.td3 import experiment as td3_experiment
import torch
#####################################
#####################################
#####################################




tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def save_evals(save_path, setting, avgs, stds):
    np.save(f'{save_path}_{setting}_avgs', avgs)
    np.save(f'{save_path}_{setting}_stds', stds)


@hydra.main(config_path='cfg', config_name='config_us')
def main(config):


    # ==============================================
    #   Define and configure
    # ==============================================
    dumper = Dumper(config.name)

    # Set plot settings
    neatplot.set_style()
    neatplot.update_rc('figure.dpi', 120)
    neatplot.update_rc('text.usetex', False)

    # Set random seed
    seed = config.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Check fixed_start_obs and num_samples_mc compatability
    assert (not config.fixed_start_obs) or config.num_samples_mc == 1, f"Need to have a fixed start obs ({config.fixed_start_obs}) or only 1 mc sample ({config.num_samples_mc})"  # NOQA

    # Set black-box functions
    env = gym.make(config.env.name)
    env.seed(seed)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    plan_env = gym.make(config.env.name)
    plan_env.seed(seed)

    # set plot fn
    plot_fn = partial(plotters[config.env.name], env=plan_env)

    reward_function = envs.reward_functions[config.env.name] if not config.alg.learn_reward else None
    #if config.normalize_env:
    #    env = NormalizedEnv(env)
    #    plan_env = NormalizedEnv(plan_env)
    #    if reward_function is not None:
    #        reward_function = make_normalized_reward_function(plan_env, reward_function)
    #    plot_fn = make_normalized_plot_fn(plan_env, plot_fn)
    if config.alg.learn_reward:
        f = get_f_batch_mpc_reward(plan_env, use_info_delta=config.teleport)
    else:
        f = get_f_batch_mpc(plan_env, use_info_delta=config.teleport)
    update_fn = make_update_obs_fn(env, teleport=config.teleport)

    # Set start obs
    start_obs = env.reset() if config.fixed_start_obs else None
    logging.info(f"Start obs: {start_obs}")

    # Set domain
    low = np.concatenate([env.observation_space.low, env.action_space.low])
    high = np.concatenate([env.observation_space.high, env.action_space.high])
    domain = [elt for elt in zip(low, high)]

    # Set algorithm
    algo_class = MPC
    algo_params = dict(
            start_obs=start_obs,
            env=plan_env,
            reward_function=reward_function,
            project_to_domain=False,
            base_nsamps=config.mpc.nsamps,#config/env/env.yaml--mpc
            planning_horizon=config.mpc.planning_horizon,
            n_elites=config.mpc.n_elites,
            beta=config.mpc.beta,
            gamma=config.mpc.gamma,
            xi=config.mpc.xi,
            num_iters=config.mpc.num_iters,
            actions_per_plan=config.mpc.actions_per_plan,
            domain=domain,
            action_lower_bound=env.action_space.low,
            action_upper_bound=env.action_space.high,
            crop_to_domain=config.crop_to_domain,
            update_fn=update_fn,
    )
    algo = algo_class(algo_params)

    # Set initial data
    data = Namespace()
    data.x = unif_random_sample_domain(domain, config.num_init_data)
    data.y = f(data.x)
    for x, y in zip(data.x, data.y):
        dumper.add('x', x)
        dumper.add('y', y)

    # Plot initial data
    ax_obs_init, fig_obs_init = plot_fn(path=None, domain=domain)
    x_obs = [xi[0] for xi in data.x]
    y_obs = [xi[1] for xi in data.x]
    if ax_obs_init:
        ax_obs_init.plot(x_obs, y_obs, 'o', color='k', ms=1)
        neatplot.save_figure(str(dumper.expdir / 'mpc_obs_init'), 'png', fig=fig_obs_init)

    # Make a test set for model evalution separate from the controller
    test_data = Namespace()
    test_data.x = unif_random_sample_domain(domain, config.test_set_size)
    test_data.y = f(test_data.x)

    # Set model
    gp_params = {
        'ls': config.env.gp.ls,
        'alpha': config.env.gp.alpha,
        'sigma': config.env.gp.sigma,
        'n_dimx': obs_dim + action_dim
    }
    if config.env.gp.periodic:
        gp_params['kernel_str'] = 'rbf_periodic'
        gp_params['periodic_dims'] = env.periodic_dimensions
        gp_params['period'] = config.env.gp.period
    multi_gp_params = {'n_dimy': obs_dim, 'gp_params': gp_params}
    gp_model_class = BatchMultiGpfsGp

    # Set acqfunction
    acqfn_params = {'n_path': config.n_paths, 'crop': True}
    acqfn_class = MultiBaxAcqFunction if not config.alg.uncertainty_sampling else UncertaintySamplingAcqFunction

    # ==============================================
    #   Computing groundtruth trajectories
    # ==============================================
    # Instantiate true algo and axes/figures
    true_algo = algo_class(algo_params)
    ax_gt, fig_gt = None, None

    # Compute and plot true path (on true function) multiple times
    full_paths = []
    true_paths = []
    returns = []
    path_lengths = []
    test_mpc_data = Namespace(x=[], y=[])
    pbar = trange(1)
    for i in pbar:
        # Run algorithm and extract paths
        full_path, output = true_algo.run_algorithm_on_f(f)
        full_paths.append(full_path)
        path_lengths.append(len(full_path.x))
        true_path = true_algo.get_exe_path_crop()
        true_paths.append(true_path)

        # Extract fraction of planning data for test_mpc_data
        true_planning_data = list(zip(true_algo.exe_path.x, true_algo.exe_path.y))
        test_points = random.sample(true_planning_data, int(config.test_set_size/config.num_eval_trials))
        new_x = [test_pt[0] for test_pt in test_points]
        new_y = [test_pt[1] for test_pt in test_points]
        test_mpc_data.x.extend(new_x)
        test_mpc_data.y.extend(new_y)

        # Plot groundtruth paths and print info
        ax_gt, fig_gt = plot_fn(true_path, ax_gt, fig_gt, domain, 'samp')
        returns.append(compute_return(output[2], 1))
        stats = {"Mean Return": np.mean(returns), "Std Return:": np.std(returns)}
        pbar.set_postfix(stats)

    # Log and dump
    returns = np.array(returns)
    dumper.add('GT Returns', returns)
    path_lengths = np.array(path_lengths)
    logging.info(f"GT Returns: returns{returns}")
    logging.info(f"GT Returns: mean={returns.mean()} std={returns.std()}")
    logging.info(f"GT Execution: path_lengths.mean()={path_lengths.mean()} path_lengths.std()={path_lengths.std()}")
    all_x = []
    for fp in full_paths:
        all_x += fp.x
    all_x = np.array(all_x)
    print(f"all_x.shape = {all_x.shape}")
    print(f"all_x.min(axis=0) = {all_x.min(axis=0)}")
    print(f"all_x.max(axis=0) = {all_x.max(axis=0)}")
    print(f"all_x.mean(axis=0) = {all_x.mean(axis=0)}")
    print(f"all_x.var(axis=0) = {all_x.var(axis=0)}")

    # Save groundtruth paths plot
    if fig_gt:
        neatplot.save_figure(str(dumper.expdir / 'mpc_gt'), 'png', fig=fig_gt)
    algo.initialize()
    mpc_policy_tru=partial(algo.execute_mpc, f=get_f_batch_mpc(plan_env, use_info_delta=config.teleport))
    video_path = f"./perturbed_videos/"
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    i = config.env.name.find('-')
    if config.env.name=='pilcocartpole-v0' or config.env.name=='lavapath-v0' or config.env.name=='bacreacher-v0':
        perturbed_env = f'{config.env.name[:i]}perturbed{config.env.name[i:]}'
    else:
        perturbed_env = f'{config.env.name[:i]}Perturbed{config.env.name[i:]}'
     # make environment and identify settings to be perturbed
    env = gym.make(perturbed_env)
    #if config.normalize_env:
    #    env = NormalizedEnv(env)
    # get the action type
    env_action_type = get_action_type(env.action_space)
   
    nsamples=0

    if isinstance(env.observation_space, spaces.Discrete):
        state_dim = 1
    else:
        state_dim = env.observation_space.shape[0]
    if config.policy=='mpc':
        policy=mpc_policy_tru
    elif config.policy=='sac':
        policy_path = f'/models/{config.policy}_{config.env.name}_{nsamples}'#change paths accordingly
        policy = SAC.load(policy_path, device='cuda').predict
    else:
        M=128
        sac_policy = TanhGaussianPolicy(obs_dim=state_dim,action_dim=action_dim,hidden_sizes=[M, M],)
        policy_path = f'/models/{config.env.name}_best_eval_policy.pt'#change paths accordingly
        sac_policy.load_state_dict(torch.load(policy_path))
        sac_policy=sac_policy
        policy=sac_policy.get_action
    #===========================
    #wandb init
    #===========================
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"robust rl {perturbed_env}_2",name=f'c_{config.policy}true',
        
        # track hyperparameters and run metadata
        config={
       
        "nsamples": nsamples,
        "type": config.policy,
        }
    )

    if not os.path.exists("./perturbed_results"):
        os.makedirs("./perturbed_results")
    save_path = f'{config.policy}_{perturbed_env}_{nsamples}'

    


    # evaluation attributes
    eval_episodes = config.num_eval_trials
    hard=True

    if eval_episodes > 20 :
        save_path=save_path+str(eval_episodes)

    ######################################################
    ####--------------PendulumPerturbed-v1-----------#####
    ######################################################
    if config.env.name == 'bacpendulum-v0':
        #ps_fm = np.arange(-0.8, 4.0, 0.1)
        ps_g = np.arange(-1.0, 3.0, 0.1)
        ps_len = np.arange(-0.8, 4.0, 0.1)
        settings = [ 'gravity', 'length']
        
        env_hor=env.horizon
        init_angle_mag=config.init_angle_mag
        if init_angle_mag != 0.2:
            save_path = f'{config.policy}_{perturbed_env}_{nsamples}_init_angle_mag_{init_angle_mag}'
        wandb.config['init_angle_mag']=init_angle_mag
        # perturb 'gravity'
        setting = 'gravity'
        avgs = []
        stds = []
        for p in ps_g:
            env.reset()
            gravity = env.g * (1 + p)
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(gravity=gravity, 
                                            seed = np.random.randint(10000),
                                            init_angle_mag=init_angle_mag), False
                else:
                    state, done = env.reset(gravity=gravity,
                                            init_angle_mag=init_angle_mag), False
                episode_reward = 0.0
                i=0
                while not done:
                    #print(state)
                    action = policy(state)
                    #print(action)
                    if isinstance(action,tuple):#SAC outputs a tuple with action as first element
                        action=action[0]
                        #print(action,'from policy')
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    i=i+1
                    if i>=env_hor:
                        done=True
                #print(i)    
                # episode done
                rewards.append(episode_reward)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' gravity with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
            wandb.log({"avg_rew_g": avg_reward, "std_rew_g": np.std(rewards), "gravity pertubation":p})
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'length'
        setting = 'length'
        avgs = []
        stds = []
        ps = np.arange(-0.8, 4.0, 0.1)
        for p in ps_len:
            env.reset()
            length = env.l * (1 + p)
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(length=length, 
                                            seed = np.random.randint(10000),
                                            init_angle_mag=init_angle_mag), False
                else: 
                    state, done = env.reset(gravity=gravity,
                                            init_angle_mag=init_angle_mag), False
                episode_reward = 0.0
                i=0
                while not done:
                    action = policy(state)
                    if isinstance(action,tuple):#SAC outputs a tuple with action as first element
                        action=action[0]
                        #print(action,'from policy')
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    i=i+1
                    if i>=env_hor:
                        done=True
                # episode done
                rewards.append(episode_reward)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' length with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
            wandb.log({"avg_rew_l": avg_reward, "std_rew_l": np.std(rewards), "length pertubation":p})
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb actions
        setting = 'action'
        avgs = []
        stds = []
        es = np.arange(0,1.1,0.1)
        for e in es:
            env.reset()
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000),init_angle_mag=init_angle_mag), False
                else:
                    state, done = env.reset(), False
                episode_reward = 0.0
                i=0
                while not done:
                    if np.random.binomial(n=1, p=e):
                        action = env.action_space.sample()
                    else: # else we use policy
                        action = policy(state)
                        if isinstance(action,tuple):#SAC outputs a tuple with action as first element
                            action=action[0]
                            #print(action,'from policy')
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    i=i+1
                    if i>=env_hor:
                        done=True
                # episode done
                rewards.append(episode_reward)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' action with e {e}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
            wandb.log({"avg_rew_a": avg_reward, "std_rew_a": np.std(rewards), "action pertubation":p})
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
    #wandb.finish()
    ######################################################
    ####--------------Lavapthperturbed-v0-----------#####
    ######################################################
    if config.env.name == 'lavapath-v0':
        ps_m = np.arange(-0.9, 2.0, 0.1)
        ps_gd = np.arange(-0.9, 2.0, 0.1)
        ps_lpb = np.arange(-0.9, 2.0, 0.1)
        settings = ['mass', 'goal_delta', 'lava_pit_boundary']

        env_hor=env.horizon
        # perturb 'mass'
        setting = 'mass'
        avgs = []
        stds = []
        for p in ps_m:
            env.reset()
            mass = env.mass * (1 + p)
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(mass=mass, 
                                            seed = np.random.randint(10000)), False
                else: 
                    state, done = env.reset(mass=mass), False
                episode_reward = 0.0
                t=0
                while not done:
                    
                    action = policy(state)
                    #print(state,action)
                    if np.isnan(action).any():
                        print(state,action)
                        #raise KeyError
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    t=t+1
                    if t>=env_hor:
                        done=True
                        #print(t)
                # episode done
                rewards.append(episode_reward)
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' mass with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            wandb.log({"avg_rew_m": avg_reward, "std_rew_m": np.std(rewards), "mass pertubation":p})
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'goal_delta'
        setting = 'goal_delta'
        avgs = []
        stds = []
        for p in ps_gd:
            env.reset()
            goal_delta = env.goal_delta * (1 + p)
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(goal_delta=goal_delta, 
                                            seed = np.random.randint(10000)), False
                else:
                    state, done = env.reset(goal_delta=goal_delta), False
                episode_reward = 0.0
                t=0
                while not done:
                    action = policy(state)
                    if np.isnan(action).any():
                        print(state,action)
                        episode_reward += (env_hor-t)*(-1)
                        break
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    t=t+1
                    if t>=env_hor:
                        done=True
                        #print(t)
                    #print(episode_reward)
                # episode done
                rewards.append(episode_reward)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' goal_delta with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            wandb.log({"avg_rew_gd": avg_reward, "std_rew_gd": np.std(rewards), "Goal Delta pertubation":p})
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'length'
        setting = 'lava_pit_boundary'
        avgs = []
        stds = []
        for p in ps_lpb:
            env.reset()
            lava_pit_boundary = env.lava_pit_boundary * (1 + p)
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(lava_pit_boundary=lava_pit_boundary, 
                                            seed = np.random.randint(10000)), False
                else: 
                    state, done = env.reset(lava_pit_boundary=lava_pit_boundary), False
                episode_reward = 0.0
                t=0
                while not done:
                    action = policy(state)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    t=t+1
                    if t>=env_hor:
                        done=True
                        #print(t)
                # episode done
                rewards.append(episode_reward)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' lava_pit_boundary with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            wandb.log({"avg_rew_lpb": avg_reward, "std_rew_lpb": np.std(rewards), "Lava_pit_boundary pertubation":p})
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
         # perturb actions
        setting = 'action'
        avgs = []
        stds = []
        es = np.arange(0,1.1,0.1)
        for e in es:
            env.reset()
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000)), False
                else:
                    state, done = env.reset(), False
                episode_reward = 0.0
                t=0
                while not done:
                    if np.random.binomial(n=1, p=e):
                        action = env.action_space.sample()
                        #print(action)
                    else: # else we use policy
                        action = policy(state)
                    if np.isscalar(action) == False:
                        action = action[0]
                    #print(action)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    #print(episode_reward)
                    t=t+1
                    if t>=env_hor:
                        done=True
                        #print(t)
                # episode done
                rewards.append(episode_reward)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' action with e {e}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            wandb.log({"avg_rew_a": avg_reward, "std_rew_a": np.std(rewards), "Action pertubation":p})
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        ######################################################
    ####--------------PilcosCartPolePerturbed-v0-----------#####
    ######################################################
    print(config.env.name)
    if config.env.name == 'pilcocartpole-v0':
        print('hello')
        print(config.env.name)
        ps_fm = np.arange(-0.8, 4.0, 0.1)
        ps_g = np.arange(-1.0, 3.0, 0.1)
        #ps_g = np.arange(2.0, 3.0, 0.1)
        ps_len = np.arange(-0.8, 4.0, 0.1)
        settings = ['force_mag', 'gravity', 'length']
        
        init_angle_mag=0.05
        wandb.config['init_angle_mag']=init_angle_mag
        env_hor=env.horizon
         # perturb 'gravity'
        setting = 'gravity'
        avgs = []
        stds = []
        for p in ps_g:
            env.reset()
            gravity = env.g * (1 + p)
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(gravity=gravity, 
                                            seed = np.random.randint(10000),
                                            init_angle_mag=init_angle_mag), False
                else:
                    state, done = env.reset(gravity=gravity,
                                            init_angle_mag=init_angle_mag), False
                episode_reward = 0.0
                t=0
                while not done:
                    action = policy(state)
                    #print(action)
                    if np.isnan(action).any():
                        print(state,action)
                        episode_reward += (env_hor-t)*(-1)
                        break
                    state, reward, done, _ = env.step(action)
                    if np.isnan(reward).any():
                        print(reward)
                        reward=-1
                    episode_reward += reward
                    t=t+1
                    if t>=env_hor:
                        done=True
                        #print(t)
                # episode done
                rewards.append(episode_reward)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' gravity with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
            wandb.log({"avg_rew_g": avg_reward, "std_rew_g": np.std(rewards), "Gravity pertubation":p})
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        # perturb actions

        setting = 'action'
        avgs = []
        stds = []
        es = np.arange(0,1.1,0.1)
        for e in es:
            env.reset()
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(seed=np.random.randint(10000),init_angle_mag=init_angle_mag), False
                else:
                    state, done = env.reset(init_angle_mag=init_angle_mag), False
                episode_reward = 0.0
                t=0
                while not done:
                    if np.random.binomial(n=1, p=e):
                        action = env.action_space.sample()
                    else: # else we use policy
                        action = policy(state)
                    if np.isscalar(action) == True:
                        action = np.array([action])
                    #print(action)
                    if np.isnan(action).any():
                        print(state,action)
                        episode_reward += (env_hor-t)*(-1)
                        break
                    state, reward, done, _ = env.step(action)
                    if np.isnan(reward).any():
                        print(reward)
                        reward=-1
                    episode_reward += reward
                    t=t+1
                    if t>=env_hor:
                        done=True
                # episode done
                rewards.append(episode_reward)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' action with e {e}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
            wandb.log({"avg_rew_a": avg_reward, "std_rew_a": np.std(rewards), "Action pertubation":e})
        # all p's are done
        save_evals(save_path, setting, avgs, stds)

        # perturb 'force_mag'
        setting = 'force_mag'
        avgs = []
        stds = []
        for p in ps_fm:
            env.reset()
            force_mag = env.force_mag * (1 + p)
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(force_mag=force_mag, 
                                            seed = np.random.randint(10000),
                                            init_angle_mag=init_angle_mag), False
                else: 
                    state, done = env.reset(force_mag=force_mag,
                                            init_angle_mag=init_angle_mag), False
                episode_reward = 0.0
                t=0
                while not done:
                    #print(state)
                    action = policy(state)
                    #print(state,action)
                    if np.isnan(action).any():
                        print(state,action)
                        episode_reward += (env_hor-t)*(-1)
                        break
                        #raise KeyError
                    state, reward, done, _ = env.step(action)
                    if np.isnan(reward).any():
                        print(reward)
                        reward=-1
                    episode_reward += reward
                    t=t+1
                    if t>=env_hor:
                        done=True
                        #print(t)
                # episode done
                rewards.append(episode_reward)
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' force_mag with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
            wandb.log({"avg_rew_f": avg_reward, "std_rew_f": np.std(rewards), "force_mag pertubation":p})
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
        
        # perturb 'length'
        setting = 'length'
        avgs = []
        stds = []
        ps = np.arange(-0.8, 4.0, 0.1)
        for p in ps_len:
            env.reset()
            length = env.l * (1 + p)
            rewards = []
            for _ in range(eval_episodes):
                # complete random environment for each episode
                if hard:
                    state, done = env.reset(length=length, 
                                            seed = np.random.randint(10000),
                                            init_angle_mag=init_angle_mag), False
                else: 
                    state, done = env.reset(gravity=gravity,
                                            init_angle_mag=init_angle_mag), False
                episode_reward = 0.0
                t=0
                while not done:
                    action = policy(state)
                    if np.isnan(action).any():
                        print(state,action)
                        episode_reward += (env_hor-t)*(-1)
                        break
                    
                    state, reward, done, _ = env.step(action)
                    if np.isnan(reward).any():
                        print(reward)
                        reward=-1
                    episode_reward += reward
                    t=t+1
                    if t>=env_hor:
                        done=True
                # episode done
                rewards.append(episode_reward)
                
            # episodes for current p are done
            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' length with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
            wandb.log({"avg_rew_l": avg_reward, "std_rew_l": np.std(rewards), "Length pertubation":p})
        # all p's are done
        save_evals(save_path, setting, avgs, stds)
         
    ######################################################
    ####-------------BACReacherPerturbed-v0--------------#####
    ######################################################
    if config.env.name == 'bacreacher-v0':
        print(config.env.name)
        env_hor=env.horizon
        settings = ['gravity', 'joint_stiffness_x', 'joint_stiffness_y', 'actuator_ctrlrange', 'joint_frictionloss']
        springref = config.springref
        ps_g = np.arange(-3.0, 3.0, 0.05)
        xs_Xjntstif = np.arange(0.0, 100.0, 2.5)
        xs_Yjntstif = np.arange(0.0, 100.0, 5.0)
        #xs_ftjntstif = np.arange(0.0, 25.0, 2.5)
        xs_act = np.arange(0.25, 1.025, 0.025)
        ps_damp = np.arange(0.0, 5.0, 0.1)
        xs_fric = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        es = np.arange(0,0.95,0.05)

        if config.eval_setting=='all':


            # perturb 'gravity'
            setting = 'gravity'
            avgs = []
            stds = []
            for p in ps_g:
                env.reset()
                gravity = env.gravity * (1 + p)
                rewards = []
                for i in range(eval_episodes):
                    #print('hello')
                    if i == 0: # save video
                        images = []
                        img = env.render(mode='rgb_array')
                        curr_videopath = f'{video_path}/{setting}_{p:.2f}.gif'
                    if hard:
                        state, done = env.reset(seed=np.random.randint(10000),
                                                gravity=gravity), False
                    else:
                        state, done = env.reset(gravity=gravity), False
                    episode_reward = 0.0
                    t=0
                    while not done:
                        if i == 0: # save video
                            images.append(img)
                        
                        #action = predict(policy, state, env_action_type)
                        action = policy(state)
                        #print(action)
                        if isinstance(action,tuple):#PPO outputs a tuple with action as first element
                            action=action[0]
                        state, reward, done, _ = env.step(action)
                        episode_reward += reward
                        t=t+1
                        if t>=env_hor:
                            done=True
                        if i == 0: # save video
                            img = env.render(mode='rgb_array')
                    # episode done
                    rewards.append(episode_reward)
                    if i == 0: # save video
                        imageio.mimsave(curr_videopath, 
                                        [np.array(img) for i, 
                                        img in enumerate(images) if i%2==0],
                                        fps=29)
                    
                # episodes for current p are done
                avg_reward = np.sum(rewards) / eval_episodes
                print("---------------------------------------")
                print(f' gravity with p {p}')
                print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
                print("---------------------------------------")
                avgs.append(avg_reward)
                stds.append(np.std(rewards))#
                wandb.log({"avg_rew_g": avg_reward, "std_rew_g": np.std(rewards), "gravity pertubation":p})
            # all p's are done
            save_evals(save_path, setting, avgs, stds)
        
        if config.eval_setting=='all' or config.eval_setting=='joint_stiffness_x':
            
            # perturb 'thigh_joint_stiffness'
            setting = 'joint_stiffness_x'
            avgs = []
            stds = []
            for x in xs_Xjntstif:
                rewards = []
                for i in range(eval_episodes):
                    if i == 0: # save video
                        images = []
                        img = env.render(mode='rgb_array')
                        curr_videopath = f'{video_path}/{setting}_{x:.2f}.gif'
                    if hard:
                        state, done = env.reset(seed=np.random.randint(10000),
                                                springref=springref,
                                                joint_stiffness_x=x), False
                    else:
                        state, done = env.reset(springref=springref,
                                                joint_stiffness_x=x), False
                    episode_reward = 0.0
                    t=0
                    while not done:
                        if i == 0: # save video
                            images.append(img)
                            
                        #action = predict(policy, state, env_action_type)
                        action = policy(state)
                        if isinstance(action,tuple):#PPO outputs a tuple with action as first element
                            action=action[0]
                        state, reward, done, _ = env.step(action)
                        episode_reward += reward
                        t=t+1
                        if t>=env_hor:
                            done=True
                        if i == 0: # save video
                            img = env.render(mode='rgb_array')
                    # episode done
                    rewards.append(episode_reward)
                    if i == 0: # save video
                        imageio.mimsave(curr_videopath, 
                                        [np.array(img) for i, 
                                        img in enumerate(images) if i%2==0],
                                        fps=29)      
                # episodes for current p are done
                avg_reward = np.sum(rewards) / eval_episodes
                print("---------------------------------------")
                print(f' x joint stiffness with x {x}')
                print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
                print("---------------------------------------")
                avgs.append(avg_reward)
                stds.append(np.std(rewards))
                wandb.log({"avg_rew_x_jt_st": avg_reward, "std_rew_x_jt_st": np.std(rewards), "x joint stiffness pertubation":x})
            # all p's are done
            save_evals(save_path, setting, avgs, stds)
        
        if config.eval_setting=='all':
            # perturb 'leg_joint_stiffness'
            setting = 'joint_stiffness_y'
            avgs = []
            stds = []
            for x in xs_Yjntstif:
                rewards = []
                for i in range(eval_episodes):
                    if i == 0: # save video
                        images = []
                        img = env.render(mode='rgb_array')
                        curr_videopath = f'{video_path}/{setting}_{x:.2f}.gif'
                    if hard:
                        state, done = env.reset(seed=np.random.randint(10000),
                                                springref=springref,
                                                joint_stiffness_y=x), False
                    else:
                        state, done = env.reset(springref=springref,
                                                joint_stiffness_y=x), False
                    episode_reward = 0.0
                    t=0
                    while not done:
                        if i == 0: # save video
                            images.append(img)
                        
                        #action = predict(policy, state, env_action_type)
                        action = policy(state)
                        if isinstance(action,tuple):#PPO outputs a tuple with action as first element
                            action=action[0]
                        state, reward, done, _ = env.step(action)
                        episode_reward += reward
                        t=t+1
                        if t>=env_hor:
                            done=True
                        if i == 0: # save video
                            img = env.render(mode='rgb_array')
                    # episode done
                    rewards.append(episode_reward)
                    if i == 0: # save video
                        imageio.mimsave(curr_videopath, 
                                        [np.array(img) for i, 
                                        img in enumerate(images) if i%2==0],
                                        fps=29)      
                # episodes for current p are done
                avg_reward = np.sum(rewards) / eval_episodes
                print("---------------------------------------")
                print(f' y joint stiffness with x {x}')
                print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
                print("---------------------------------------")
                avgs.append(avg_reward)
                stds.append(np.std(rewards))
                wandb.log({"avg_rew_y_jt_st": avg_reward, "std_rew_y_jt_st": np.std(rewards), "y joint stiffness pertubation":x})
            
            # all p's are done
            save_evals(save_path, setting, avgs, stds)
        
        if config.eval_setting=='all':

            # perturb actuator control range
            setting = 'actuator_ctrlrange'
            avgs = []
            stds = []
            for x in xs_act:
                rewards = []
                for i in range(eval_episodes):
                    if i == 0: # save video
                        images = []
                        img = env.render(mode='rgb_array')
                        curr_videopath = f'{video_path}/{setting}_{x:.2f}.gif'
                    if hard:
                        state, done = env.reset(seed=np.random.randint(10000),
                                                actuator_ctrlrange=(-x,x)), False
                    else:
                        state, done = env.reset(actuator_ctrlrange=(-x,x)), False
                    episode_reward = 0.0
                    t=0
                    while not done:
                        if i == 0: # save video
                            images.append(img)
                            
                        #action = predict(policy, state, env_action_type)
                        action = policy(state)
                        if isinstance(action,tuple):#PPO outputs a tuple with action as first element
                            action=action[0]
                        state, reward, done, _ = env.step(action)
                        episode_reward += reward
                        t=t+1
                        if t>=env_hor:
                            done=True
                        if i == 0: # save video
                            img = env.render(mode='rgb_array')
                    # episode done
                    rewards.append(episode_reward)
                    if i == 0: # save video
                        imageio.mimsave(curr_videopath, 
                                        [np.array(img) for i, 
                                        img in enumerate(images) if i%2==0],
                                        fps=29)      
                # episodes for current p are done
                avg_reward = np.sum(rewards) / eval_episodes
                print("---------------------------------------")
                print(f' actuator ctrlrange is [-{x}, {x}]')
                print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
                print("---------------------------------------")
                avgs.append(avg_reward)
                stds.append(np.std(rewards))
                wandb.log({"avg_rew_act_ctrl": avg_reward, "std_rew_act_ctrl": np.std(rewards), "Actuator Ctrlrange pertubation":x})
            
            # all p's are done
            save_evals(save_path, setting, avgs, stds)       
        
        if config.eval_setting=='all':

            # perturb 'joint_damping'
            setting = 'joint_damping'
            avgs = []
            stds = []
            for p in ps_damp:
                env.reset()
                rewards = []
                for i in range(eval_episodes):
                    if i == 0: # save video
                        images = []
                        img = env.render(mode='rgb_array')
                        curr_videopath = f'{video_path}/{setting}_{p:.2f}.gif'
                    if hard:
                        state, done = env.reset(seed=np.random.randint(10000),
                                                joint_damping_p=p), False
                    else:
                        state, done = env.reset(joint_damping_p=p), False
                    episode_reward = 0.0
                    t=0
                    while not done:
                        if i == 0: # save video
                            images.append(img)
                        
                        #action = predict(policy, state, env_action_type)
                        action = policy(state)
                        if isinstance(action,tuple):#PPO outputs a tuple with action as first element
                            action=action[0]
                        state, reward, done, _ = env.step(action)
                        episode_reward += reward
                        t=t+1
                        if t>=env_hor:
                            done=True
                        if i == 0: # save video
                            img = env.render(mode='rgb_array')
                    # episode done
                    rewards.append(episode_reward)
                    if i == 0: # save video
                        imageio.mimsave(curr_videopath, 
                                        [np.array(img) for i, 
                                        img in enumerate(images) if i%2==0],
                                        fps=29)
                    
                # episodes for current p are done
                avg_reward = np.sum(rewards) / eval_episodes
                print("---------------------------------------")
                print(f' joint damping with p {p}')
                print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
                print("---------------------------------------")
                avgs.append(avg_reward)
                stds.append(np.std(rewards))
                wandb.log({"avg_rew_jt_dp": avg_reward, "std_rew_jt_dp": np.std(rewards), "joint damping pertubation":p})
            # all p's are done
            save_evals(save_path, setting, avgs, stds)
        
        if config.eval_setting=='all':

            # perturb 'joint_frictionloss'
            setting = 'joint_frictionloss'
            avgs = []
            stds = []
            for x in xs_fric:
                rewards = []
                for i in range(eval_episodes):
                    if i == 0: # save video
                        images = []
                        img = env.render(mode='rgb_array')
                        curr_videopath = f'{video_path}/{setting}_{x:.2f}.gif'
                    if hard:
                        state, done = env.reset(seed=np.random.randint(10000),
                                                joint_frictionloss=x), False
                    else:
                        state, done = env.reset(joint_frictionloss=x), False
                    episode_reward = 0.0
                    t=0
                    while not done:
                        if i == 0: # save video
                            images.append(img)
                            
                        #action = predict(policy, state, env_action_type)
                        action = policy(state)
                        if isinstance(action,tuple):#PPO outputs a tuple with action as first element
                            action=action[0]
                        state, reward, done, _ = env.step(action)
                        episode_reward += reward
                        t=t+1
                        if t>=env_hor:
                            done=True
                        if i == 0: # save video
                            img = env.render(mode='rgb_array')
                    # episode done
                    rewards.append(episode_reward)
                    if i == 0: # save video
                        imageio.mimsave(curr_videopath, 
                                        [np.array(img) for i, 
                                        img in enumerate(images) if i%2==0],
                                        fps=29)      
                # episodes for current p are done
                avg_reward = np.sum(rewards) / eval_episodes
                print("---------------------------------------")
                print(f'joint frictionloss with x {x}')
                print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
                print("---------------------------------------")
                avgs.append(avg_reward)
                stds.append(np.std(rewards))
                wandb.log({"avg_rew_jt_fl": avg_reward, "std_rew_jt_fl": np.std(rewards), "joint frictionloss pertubation":x})
            
            # all p's are done
            save_evals(save_path, setting, avgs, stds)
        
        if config.eval_setting=='all':

            # perturb actions
            setting = 'action'
            avgs = []
            stds = []
            for e in es:
                env.reset()
                rewards = []
                for i in range(eval_episodes):
                    if i == 0: # save video
                        images = []
                        img = env.render(mode='rgb_array')
                        curr_videopath = f'{video_path}/{setting}_{e:.2f}.gif'
                    state, done = env.reset(seed=np.random.randint(100000)), False
                    episode_reward = 0.0
                    t=0
                    while not done:
                        if i == 0: # save video
                            images.append(img)
                            
                        if np.random.binomial(n=1, p=e):
                            action = env.action_space.sample()
                        else: # else we use policy
                            
                            #action = predict(policy, state, env_action_type)
                            action = policy(state)
                            if isinstance(action,tuple):#PPO outputs a tuple with action as first element
                                action=action[0]
                        state, reward, done, _ = env.step(action)
                        episode_reward += reward
                        t=t+1
                        if t>=env_hor:
                            done=True
                        if i == 0: # save video
                            img = env.render(mode='rgb_array')
                    # episode done
                    rewards.append(episode_reward)
                    if i == 0: # save video
                        imageio.mimsave(curr_videopath, 
                                        [np.array(img) for i, 
                                        img in enumerate(images) if i%2==0],
                                        fps=29)
                # episodes for current p are done
                avg_reward = np.sum(rewards) / eval_episodes
                print("---------------------------------------")
                print(f' action with e {e}')
                print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
                print("---------------------------------------")
                avgs.append(avg_reward)
                stds.append(np.std(rewards))
                wandb.log({"avg_rew_a": avg_reward, "std_rew_a": np.std(rewards), "Action pertubation":e})
            # all p's are done
            save_evals(save_path, setting, avgs, stds)   
 

    

if __name__ == '__main__':
    main()
