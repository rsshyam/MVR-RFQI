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



tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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
    if config.normalize_env:
        env = NormalizedEnv(env)
        plan_env = NormalizedEnv(plan_env)
        if reward_function is not None:
            reward_function = make_normalized_reward_function(plan_env, reward_function)
        plot_fn = make_normalized_plot_fn(plan_env, plot_fn)
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

    # ==============================================
    #   Optionally: fit GP hyperparameters (then exit)
    # ==============================================
    if config.fit_hypers:
        # Use test_mpc_data to fit hyperparameters
        fit_data = test_mpc_data
        assert len(fit_data.x) <= 3000, "fit_data larger than preset limit (can cause memory issues)"

        logging.info('\n'+'='*60+'\n Fitting Hyperparameters\n'+'='*60)
        logging.info(f'Number of observations in fit_data: {len(fit_data.x)}')

        # Plot hyper fitting data
        ax_obs_hyper_fit, fig_obs_hyper_fit = plot_fn(path=None, domain=domain)
        x_obs = [xi[0] for xi in fit_data.x]
        y_obs = [xi[1] for xi in fit_data.x]
        if ax_obs_hyper_fit:
            ax_obs_hyper_fit.plot(x_obs, y_obs, 'o', color='k', ms=1)
            neatplot.save_figure(str(dumper.expdir / 'mpc_obs_hyper_fit'), 'png', fig=fig_obs_hyper_fit)

        # Perform hyper fitting
        for idx in range(len(data.y[0])):
            data_fit = Namespace(x=fit_data.x, y=[yi[idx] for yi in fit_data.y])
            gp_params = get_gpflow_hypers_from_data(data_fit, print_fit_hypers=True,
                                                    opt_max_iter=config.env.gp.opt_max_iter)
            logging.info(f'gp_params for output {idx} = {gp_params}')

        # End script if hyper fitting bc need to include in config
        return

    # ==============================================
    #   Run main algorithm loop
    # ==============================================

    # Set current_obs as fixed start_obs or reset plan_env
    if config.alg.rollout_sampling:
        current_obs = start_obs.copy() if config.fixed_start_obs else plan_env.reset()
        current_t = 0

    posterior_returns = None
    policy_use='ppo'
    #paths=f'FQI_{perturbed_env}_d4rl'
    save_path = f'/data'
    lnt_return_avgs=[]
    lnt_return_stds=[]
    tru_return_avgs=[]
    tru_return_stds=[]
    sample_sizes=[]
    for i in range(config.num_iters):
        logging.info('---' * 5 + f' Start iteration i={i} ' + '---' * 5)
        logging.info(f'Length of data.x: {len(data.x)}')
        logging.info(f'Length of data.y: {len(data.y)}')

        # Initialize various axes and figures
        ax_all, fig_all = plot_fn(path=None, domain=domain)
        ax_postmean, fig_postmean = plot_fn(path=None, domain=domain)
        ax_samp, fig_samp = plot_fn(path=None, domain=domain)
        ax_obs, fig_obs = plot_fn(path=None, domain=domain)

        # Set model as None, instantiate when needed
        model = None

        if config.alg.use_acquisition:#cfg/alg/us--
            model = gp_model_class(multi_gp_params, data)
            # Set and optimize acquisition function
            acqfn_base = acqfn_class(params=acqfn_params, model=model, algorithm=algo)
            acqfn = MCAcqFunction(acqfn_base, {"num_samples_mc": config.num_samples_mc})
            acqopt = AcqOptimizer()
            acqopt.initialize(acqfn)
            if config.alg.rollout_sampling:
                x_test = [np.concatenate([current_obs, env.action_space.sample()]) for _ in range(config.n_rand_acqopt)]
            elif config.sample_exe and not config.alg.uncertainty_sampling:
                all_x = []
                for path in acqfn.exe_path_full_list:
                    all_x += path.x
                n_path = int(config.n_rand_acqopt * config.path_sampling_fraction)
                n_rand = config.n_rand_acqopt - n_path
                x_test = random.sample(all_x, n_path)
                x_test = np.array(x_test)
                x_test += np.random.randn(*x_test.shape) * 0.01
                x_test = list(x_test)
                x_test += unif_random_sample_domain(domain, n=n_rand)
            else:
                x_test = unif_random_sample_domain(domain, n=config.n_rand_acqopt)
            x_next, acq_val = acqopt.optimize(x_test)
            dumper.add('Acquisition Function Value', acq_val)
            dumper.add('x_next', x_next)
            logging.info(f'x_next: {x_next}')

            # Plot true path and posterior path samples
            ax_all, fig_all = plot_fn(true_path, ax_all, fig_all, domain, 'true')
            if ax_all is not None:
                # Plot observations
                x_obs, y_obs = make_plot_obs(data.x, env, config.env.normalize_env)
                ax_all.scatter(x_obs, y_obs, color='grey', s=10, alpha=0.3)
                ax_obs.plot(x_obs, y_obs, 'o', color='k', ms=1)

                # Plot execution path posterior samples
                for path in acqfn.exe_path_list:
                    ax_all, fig_all = plot_fn(path, ax_all, fig_all, domain, 'samp')
                    ax_samp, fig_samp = plot_fn(path, ax_samp, fig_samp, domain, 'samp')

                # Plot x_next
                x, y = make_plot_obs(x_next, env, config.env.normalize_env)
                ax_all.scatter(x, y, facecolors='deeppink', edgecolors='k', s=120, zorder=100)
                ax_obs.plot(x, x, 'o', mfc='deeppink', mec='k', ms=12, zorder=100)

            # Store returns of posterior samples
            posterior_returns = [compute_return(output[2], 1) for output in acqfn.output_list]
            dumper.add('Posterior Returns', posterior_returns)
        elif config.alg.use_mpc:
            model = gp_model_class(multi_gp_params, data)
            algo.initialize()

            policy = partial(algo.execute_mpc, f=make_postmean_fn(model))
            action = policy(current_obs)
            x_next = np.concatenate([current_obs, action])
        else:
            x_next = unif_random_sample_domain(domain, 1)[0]

        # ==============================================
        #   Periodically run evaluation and plot
        # ==============================================
        if  (i % config.eval_frequency == 1 and i > 10) or i + 1 == config.num_iters:
            if model is None:
                model = gp_model_class(multi_gp_params, data)
            if posterior_returns:
                logging.info(f"Current posterior returns: {posterior_returns}")
                logging.info(f"Current posterior returns: mean={np.mean(posterior_returns)}, std={np.std(posterior_returns)}")
            with Timer("Generate data using current model"):
                # execute the best we can
                # this is required to delete the current execution path
                algo.initialize()
                
                postmean_fn = make_postmean_fn(model)
                

                #arguements to be sent to gen_data_file_mod.py to generate data
                args=Namespace()
                args.env=config.env.name
                args.eps=config.data_eps #(random action probability while generating data)
                args.buffer_size=config.buffer_size #(data size)
                args.verbose=False 
                args.device= 'cuda' 
                args.gendata_pol='sac' #(policy used to generate data)
                args.mixed='False'
                args.samples=i+1
                args.normalize_env=True #normalize environment as done in this code
                verbose=args.verbose

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
                if args.device not in ['cpu', 'cuda', 'cuda:0', 'cuda:1',  'cuda:2', 'cuda:3','auto']:
                    raise NotImplementedError
                    
                # create folder inside current directory of experiment to store generated data
                if not os.path.exists("./offline_data"):
                    os.makedirs("./offline_data")
                    
                # check mixed option
                if args.mixed == 'True' or args.mixed == 'False':
                    pass
                else:
                    raise NotImplementedError
                #policy = partial(algo.execute_mpc, f=postmean_fn)
                real_returns = []
                lnt_returns = []
                #mses = []
                pbar = trange(config.num_eval_trials)

                #mpc based policy generation 
                if config.check_lrntmod_perf == True:####use mpc on learnt model to generate data
                    for j in pbar:
                        mpc_policy_lnt = partial(algo.execute_mpc, f=postmean_fn)#mpc policy on learnt model
                        ###evaluation
                        real_obs, real_actions, real_rewards = evaluate_policy(env, mpc_policy_lnt, start_obs=start_obs,mpc_pass=False)
                        real_return = compute_return(real_rewards, 1)
                        lnt_returns.append(real_return)
                        real_path = Namespace()
                        real_path.x = real_obs
                        ax_all, fig_all = plot_fn(real_path, ax_all, fig_all, domain, 'postmean')
                        ax_postmean, fig_postmean = plot_fn(real_path, ax_postmean, fig_postmean, domain, 'samp')
                        #mses.append(rollout_mse(algo.old_exe_paths[-1], f))
                        stats = {"Mean Return": np.mean(lnt_returns), "Std Return:": np.std(lnt_returns)}
                        print(stats,'lnt')
                       
                        #mpc policy on true model
                        mpc_policy_tru=partial(algo.execute_mpc, f=get_f_batch_mpc(plan_env, use_info_delta=config.teleport))
                        #evaluation
                        real_obs, real_actions, real_rewards = evaluate_policy(env, mpc_policy_tru, start_obs=start_obs,mpc_pass=False)
                        real_return = compute_return(real_rewards, 1)
                        real_returns.append(real_return)
                        real_path = Namespace()
                        real_path.x = real_obs
                        ax_all, fig_all = plot_fn(real_path, ax_all, fig_all, domain, 'postmean')
                        ax_postmean, fig_postmean = plot_fn(real_path, ax_postmean, fig_postmean, domain, 'samp')
                        #mses.append(rollout_mse(algo.old_exe_paths[-1], f))
                        stats = {"Mean Return": np.mean(real_returns), "Std Return:": np.std(real_returns)}
                        print(stats,'true')
                       
                    #generate data using learnt model
                    if config.generate_data_using_mpc==True:
                        args.gendata_pol='mpc'
                        print(args)
                        generate_dataset(args.env, args.gendata_pol, args.eps, state_dim, action_dim,args,postmean_fn, args.samples, args.normalize_env, int(args.buffer_size), verbose,mpc_func=mpc_policy_lnt)
                    
                    #generate data using true model
                    if config.generate_data_using_true_mpc==True:
                        args.gendata_pol='mpc'
                        print(args)
                        #denote number of samples as 0 to denote true model
                        generate_dataset(args.env, args.gendata_pol, args.eps, state_dim, action_dim,args,None, 0, args.normalize_env, int(args.buffer_size), verbose, mpc_func=mpc_policy_tru)

                    lnt_return_avgs.append(np.mean(lnt_returns))
                    lnt_return_stds.append(np.std(lnt_returns))
                    tru_return_avgs.append(np.mean(real_returns))
                    tru_return_stds.append(np.std(real_returns))
                    sample_sizes.append(i+1)
                    print(lnt_returns)
                    print(real_returns)

                if config.check_lrntmod_perf == False: ###to evaluate sac policy trained on true or learnt model
                    #create env based on learnt model
                    lrnt_env= gym.make(config.env.name,f_dynamics=postmean_fn)
                    if config.normalize_env:
                        lrnt_env = NormalizedEnv(lrnt_env)
                    sample_sizes.append(i+1)#no of queries to true model to construct the learnt model
                    
                    #define policy on learnt environment
                    lnt_sac_policy=SAC("MlpPolicy",lrnt_env,gamma=0.99,learning_rate=0.0013670498038933314, batch_size=2048,buffer_size= 1000000,learning_starts=10000,train_freq=16,tau= 0.02,policy_kwargs=dict(log_std_init=  -1.277524874557824,net_arch=[256, 256]),verbose=1)
                    trn_steps_scale=10

                    #log training evaluations
                    new_logger = configure("./trainin_log/", ["log"])
                    lnt_sac_policy.set_logger(new_logger)
                    
                    #train policy
                    lnt_sac_policy.learn(total_timesteps=config.training_steps*trn_steps_scale)    
                    
                    #evaluate policy
                    for j in pbar:
                        real_obs, real_actions, real_rewards = evaluate_policy(env, lnt_sac_policy.predict, start_obs=start_obs,mpc_pass=False)
                        real_return = compute_return(real_rewards, 1)
                        real_returns.append(real_return)
                        real_path = Namespace()
                        real_path.x = real_obs
                        ax_all, fig_all = plot_fn(real_path, ax_all, fig_all, domain, 'postmean')
                        ax_postmean, fig_postmean = plot_fn(real_path, ax_postmean, fig_postmean, domain, 'samp')
                        #mses.append(rollout_mse(algo.old_exe_paths[-1], f))
                        stats = {"Mean Return": np.mean(real_returns), "Std Return:": np.std(real_returns)}
                        print(stats,'lnt')
                    lnt_return_avgs.append(np.mean(real_returns))
                    lnt_return_stds.append(np.std(real_returns))

                    real_returns = []
                    
                    #define policy on true environment
                    tru_sac_policy=SAC("MlpPolicy",plan_env,gamma=0.99,learning_rate=0.0013670498038933314, batch_size=2048,buffer_size= 1000000,learning_starts=10000,train_freq=16,tau= 0.02,policy_kwargs=dict(log_std_init=  -1.277524874557824,net_arch=[256, 256]),verbose=1)
                    
                    #define logger and train policy
                    trn_steps_scale=10
                    new_logger_tru = configure("./trainin_log_tru/", ["log"])
                    tru_sac_policy.set_logger(new_logger_tru)
                    tru_sac_policy.learn(total_timesteps=config.training_steps*trn_steps_scale)
                    
                    #evaluate policy
                    for j in pbar:
                        real_obs, real_actions, real_rewards = evaluate_policy(env, tru_sac_policy.predict, start_obs=start_obs,mpc_pass=False)
                        real_return = compute_return(real_rewards, 1)
                        real_returns.append(real_return)
                        real_path = Namespace()
                        real_path.x = real_obs
                        ax_all, fig_all = plot_fn(real_path, ax_all, fig_all, domain, 'postmean')
                        ax_postmean, fig_postmean = plot_fn(real_path, ax_postmean, fig_postmean, domain, 'samp')
                        #mses.append(rollout_mse(algo.old_exe_paths[-1], f))
                        stats = {"Mean Return": np.mean(real_returns), "Std Return:": np.std(real_returns)}
                        print(stats,'tru')
                    tru_return_avgs.append(np.mean(real_returns))
                    tru_return_stds.append(np.std(real_returns))
                    
                if config.check_true_sac_perf == True: #to evaluate and generate data using SAC trained on true model (similar to previous)
                    sac_policy=SAC("MlpPolicy", plan_env,learning_rate=1e-3)
                    sac_policy.learn(total_timesteps=config.training_steps*2)
                    for j in pbar:
                        real_obs, real_actions, real_rewards = evaluate_policy(env, sac_policy.predict, start_obs=start_obs,mpc_pass=False)
                        real_return = compute_return(real_rewards, 1)
                        real_returns.append(real_return)
                        real_path = Namespace()
                        real_path.x = real_obs
                        ax_all, fig_all = plot_fn(real_path, ax_all, fig_all, domain, 'postmean')
                        ax_postmean, fig_postmean = plot_fn(real_path, ax_postmean, fig_postmean, domain, 'samp')
                        stats = {"Mean Return": np.mean(real_returns), "Std Return:": np.std(real_returns)}
                        print(stats,'true')
                    tru_return_avgs.append(np.mean(real_returns))
                    tru_return_stds.append(np.std(real_returns))
                    sac_policy.save(f"sac_{config.env.name}_{0}")
                    if config.generate_data_using_true_sac==True:#change in config_us to generate data
                        #sac_policy.save(f"sac_{config.env.name}_{0}")#0 to denote it is true model and not learnt model
                        generate_dataset(args.env, args.gendata_pol, args.eps, state_dim, action_dim,args,None, 0, args.normalize_env, int(args.buffer_size), verbose)


                if config.check_sac_perf == True:  #to evaluate and generate data using SAC trained on learnt model (similar to previous)
                    lrnt_env= gym.make(config.env.name,f_dynamics=postmean_fn)
                    if config.normalize_env:
                        lrnt_env = NormalizedEnv(lrnt_env)
                    sac_policy_lnt=SAC("MlpPolicy", lrnt_env,learning_rate=1e-3)
                    sac_policy_lnt.learn(total_timesteps=config.training_steps*2)
                    sample_sizes.append(i+1)
                    for j in pbar:
                        real_obs, real_actions, real_rewards = evaluate_policy(env, sac_policy_lnt.predict, start_obs=start_obs,mpc_pass=False)
                        real_return = compute_return(real_rewards, 1)
                        real_returns.append(real_return)
                        real_path = Namespace()
                        real_path.x = real_obs
                        ax_all, fig_all = plot_fn(real_path, ax_all, fig_all, domain, 'postmean')
                        ax_postmean, fig_postmean = plot_fn(real_path, ax_postmean, fig_postmean, domain, 'samp')
                        stats = {"Mean Return": np.mean(real_returns), "Std Return:": np.std(real_returns)}
                        print(stats,'true')
                    lnt_return_avgs.append(np.mean(real_returns))
                    lnt_return_stds.append(np.std(real_returns))
                    if config.generate_data_using_sac==True:
                        sac_policy_lnt.save(f"sac_{config.env.name}_{i+1}")#change in config_us to generate data
                        generate_dataset(args.env, args.gendata_pol, args.eps, state_dim, action_dim,args, postmean_fn, args.samples, args.normalize_env, int(args.buffer_size), verbose)

                if config.check_sac_perf_multiple == True:#evaluate performance of SAC on learnt model with different training steps

                    lrnt_env= gym.make(config.env.name,f_dynamics=postmean_fn)
                    if config.normalize_env:
                        lrnt_env = NormalizedEnv(lrnt_env)
                    for k in range(10):
                        sac_policy_lnt=SAC("MlpPolicy", lrnt_env,learning_rate=1e-3)
                        sac_policy_lnt.learn(total_timesteps=config.training_steps*(k+1))
                        sample_sizes.append(config.training_steps*(k+1))
                        real_returns=[]
                        for j in pbar:
                            real_obs, real_actions, real_rewards = evaluate_policy(env, sac_policy_lnt.predict, start_obs=start_obs,mpc_pass=False)
                            real_return = compute_return(real_rewards, 1)
                            real_returns.append(real_return)
                            real_path = Namespace()
                            real_path.x = real_obs
                            ax_all, fig_all = plot_fn(real_path, ax_all, fig_all, domain, 'postmean')
                            ax_postmean, fig_postmean = plot_fn(real_path, ax_postmean, fig_postmean, domain, 'samp')
                            stats = {"Mean Return": np.mean(real_returns), "Std Return:": np.std(real_returns)}
                        lnt_return_avgs.append(np.mean(real_returns))
                        lnt_return_stds.append(np.std(real_returns))
                        print(lnt_return_avgs,'avgs')
                        print(lnt_return_stds,'stds')
                        
                
                if config.check_mpc_perf==True: #evaluate performance of 'mpc using learnt model'
                    #mpc_policy=partial(algo.execute_mpc, f=postmean_fn)
                    mpc_policy=partial(algo.execute_mpc, f=get_f_batch_mpc(plan_env, use_info_delta=config.teleport))
                    for j in pbar:
                        real_obs, real_actions, real_rewards = evaluate_policy(env, mpc_policy, start_obs=start_obs,mpc_pass=False)
                        real_return = compute_return(real_rewards, 1)
                        real_returns.append(real_return)
                        real_path = Namespace()
                        real_path.x = real_obs
                        ax_all, fig_all = plot_fn(real_path, ax_all, fig_all, domain, 'postmean')
                        ax_postmean, fig_postmean = plot_fn(real_path, ax_postmean, fig_postmean, domain, 'samp')
                        #mses.append(rollout_mse(algo.old_exe_paths[-1], f))
                        stats = {"Mean Return": np.mean(real_returns), "Std Return:": np.std(real_returns)}
                        print(stats,'true')
                        if config.use_neptune:
                            run["tru_returns"]=real_return
                    tru_return_avgs.append(np.mean(real_returns))
                    tru_return_stds.append(np.std(real_returns))
                    sample_sizes.append(i+1)

                algo.old_exe_paths = []
                logging.info('lnt_return_avgs: %s', lnt_return_avgs)
                logging.info('tru_return_avgs : %s', tru_return_avgs)
                logging.info('sample_sizes : %s',sample_sizes)
        



        # Query function, update data
        y_next = f([x_next])[0]
        data.x.append(x_next)
        data.y.append(y_next)
        dumper.add('x', x_next)
        dumper.add('y', y_next)
        if config.alg.rollout_sampling:
            current_t += 1
            if current_t > env.horizon:
                current_t = 0
                current_obs = start_obs.copy() if config.fixed_start_obs else plan_env.reset()
            else:
                current_obs += y_next[-obs_dim:]
        # Dumper save
        dumper.save()
        plt.close('all')
    # check path
    save_path=save_path+'/'+config.time_date
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logging.info('lnt_return_avgs: %s', lnt_return_avgs)
    logging.info('tru_return_avgs : %s', tru_return_avgs)
    logging.info('sample_sizes : %s',sample_sizes)
    np.save(f'{save_path}/{config.time_date}-{config.training_steps}_{policy_use}_{config.name}_{config.seed}_lnt_avgs', lnt_return_avgs)
    np.save(f'{save_path}/{config.time_date}-{config.training_steps}_{policy_use}_{config.name}_{config.seed}_lnt_stds', lnt_return_stds)
    np.save(f'{save_path}/{config.time_date}-{config.training_steps}_{policy_use}_{config.name}_{config.seed}_tru_avgs', tru_return_avgs)
    np.save(f'{save_path}/{config.time_date}-{config.training_steps}_{policy_use}_{config.name}_{config.seed}_tru_stds', tru_return_stds)
    np.save(f'{save_path}/{config.time_date}-{config.training_steps}_{policy_use}_{config.name}_{config.seed}_sample_sizes', sample_sizes)
if __name__ == '__main__':
    main()
