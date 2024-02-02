import argparse
import gym
from gym import spaces
import numpy as np
import os
import torch
import imageio

from fqi import PQL_BCQ
import wandb
import random
from sys import exit  


def predict(policy, state, env_action_type):
    '''
    PQL version of predict.
    '''
    action = policy.select_action(np.array(state))
    if env_action_type == 'discrete':
        return np.rint(action[0]).astype(int)
    elif env_action_type == 'continuous':
        return action[0]
    elif env_action_type == 'multi_continuous':
        return action
    else:
        raise NotImplementedError


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

def load_policy(policy, load_path, device):
    policy.actor.load(f'{load_path}_actor', device=device)
    policy.critic.load(f'{load_path}_critic', device=device)
    policy.vae.load(f'{load_path}_action_vae', device=device)
    policy.vae2.load(f'{load_path}_state_vae', device=device)   
    return policy

def save_evals(save_path, setting, avgs, stds):
    np.save(f'{save_path}_{setting}_avgs', avgs)
    np.save(f'{save_path}_{setting}_stds', stds)


def eval_FQI(state_dim, action_dim, max_state, min_action, max_action,
             hard, perturbed_env, args, eval_episodes=20):
    '''
    Evaluate FQI on perturbed environments.
    '''
    # parse paths
    print(paths)
    print(paths['load_path'])
    load_path = f"./models/{paths['load_path']}"
    save_path = f"./perturbed_results/results_reacher_springref/{paths['save_path']}_springref{args.springref}"
    video_path = f"./perturbed_videos/{paths['save_path']}_springref{args.springref}"
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    
    # evaluation attributes
    eval_episodes = args.eval_episodes
    
    # make environment and identify settings to be perturbed
    env = gym.make(perturbed_env)
    # get the action type
    env_action_type = get_action_type(env.action_space)


    #===========================
    #wandb init
    #===========================
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"robust rl {perturbed_env}_2_springref_check",name=f'{args.comment}_fqi_eps{args.data_eps}_{args.nsamples}_springref_{args.springref}',
        
        # track hyperparameters and run metadata
        config={
        "data_eps": args.data_eps,
        "comment": args.comment,
        "nsamples": args.nsamples,
        "type": 'rfqi',
        "springref": args.springref
        }
    )







    # Initialize policy
    policy = PQL_BCQ(state_dim, action_dim, max_state, min_action, max_action,
                     args.device, args.discount, args.tau, args.lmbda, args.phi,
                     n_action=args.n_action,
                     n_action_execute=args.n_action_execute,
                     backup=args.backup, ql_noise=args.ql_noise,
                     actor_lr=args.actor_lr, beta=args.beta, vmin=args.vmin)
    policy = load_policy(policy, load_path, args.device)

     ######################################################
    ####-------------BACReacherPerturbed-v0--------------#####
    ######################################################
    if args.env == 'bacreacher-v0':
        settings = ['gravity', 'joint_stiffness_x', 'joint_stiffness_y', 'actuator_ctrlrange', 'joint_frictionloss']
        springref = args.springref
        #ps_g = np.arange(-0.5, 0.0, 0.05)
        #xs_Xjntstif = np.arange(0.0, 32.5, 2.5)
        #xs_Yjntstif = np.arange(0.0, 60.0, 5.0)
        #xs_ftjntstif = np.arange(0.0, 25.0, 2.5)
        #xs_act = np.arange(0.85, 1.025, 0.025)
        #ps_damp = np.arange(0.0, 1.1, 0.1)
        #xs_fric = [0.0, 1.0, 2.0, 3.0, 4.0]
        #es = np.arange(0,0.45,0.05)


        ps_g = np.arange(-3.0, 3.0, 0.05)
        xs_Xjntstif = np.arange(0.0, 100.0, 2.5)
        xs_Yjntstif = np.arange(0.0, 100.0, 5.0)
        #xs_ftjntstif = np.arange(0.0, 25.0, 2.5)
        xs_act = np.arange(0.25, 1.025, 0.025)
        ps_damp = np.arange(0.0, 5.0, 0.1)
        xs_fric = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        es = np.arange(0,0.95,0.05)

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
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
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
                while not done:
                    if i == 0: # save video
                        images.append(img)
                        
                    action = predict(policy, state, env_action_type)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    
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
    wandb.finish()

if __name__ == "__main__":


   
    parser = argparse.ArgumentParser()
    # OpenAI gym environment name (need to be consistent with the dataset name)
    parser.add_argument("--env", default="Hopper-v2")
    # sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=1, type=int)  
    # how often (in time steps) we evaluate
    parser.add_argument("--eval_episodes", default=20, type=int)
    # evaluate on randomly initialized environment every episode
    parser.add_argument("--hard", default='True', type=str)
    # the epsilon used to generate the training data
    parser.add_argument("--data_eps", default=0.3, type=float)
    # use d4rl dataset
    parser.add_argument("--d4rl", default='False', type=str)
    parser.add_argument("--d4rl_v2", default='False', type=str)
    parser.add_argument("--d4rl_expert", default='False', type=str)
    # use mixed policy
    parser.add_argument("--mixed", default='False', type=str)
    # policy used to generate data
    parser.add_argument("--gendata_pol", default='sac', type=str)
    # check policy comment
    parser.add_argument("--comment", default='', type=str)
    # device to run evaluations
    parser.add_argument("--device", default='cuda', type=str)

    #==========================BCQ parameter==========================
    # discount factor
    parser.add_argument("--discount", default=0.99)  
    # Target network update rate
    parser.add_argument("--tau", default=0.005)  
    # Weighting for clipped double Q-learning in BCQ
    parser.add_argument("--lmbda", default=0.75)  
    # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--phi", default=0.1, type=float)
    # learning rate of actor
    parser.add_argument("--actor_lr", default=1e-3, type=float) 
    # number of sampling action for policy (in backup)
    parser.add_argument("--n_action", default=100, type=int) 
    # number of sampling action for policy (in execution)
    parser.add_argument("--n_action_execute", default=100, type=int) 

    #==========================PQL-BCQ parameter=======================
    parser.add_argument("--backup", type=str, default="QL") # "QL": q learning (Q-max) back up, "AC": actor-critic backup
    parser.add_argument("--ql_noise", type=float, default=0.15) # Noise of next action in QL
    parser.add_argument("--automatic_beta", type=str, default='True')  # If true, use percentile for b (beta is the b in paper)
    parser.add_argument("--beta_percentile", type=float, default=2.0)  # Use x-Percentile as the value of b
    parser.add_argument("--beta", default=-0.4, type=float)  # hardcoded b, only effective when automatic_beta = False
    parser.add_argument("--vmin", default=0, type=float) # min value of the environment. Empirically I set it to be the min of 1000 random rollout.
    parser.add_argument("--dtalrnt", default='False', type=str)
    parser.add_argument('--nsamples', default=3, type=int)

    parser.add_argument('--springref', default=2, type=int)
    args = parser.parse_args()

    if args.data_eps==1:
        args.data_eps=int(args.data_eps)

    if not os.path.exists("./perturbed_results"):
        os.makedirs("./perturbed_results")
    if not os.path.exists("./perturbed_videos"):
        os.makedirs("./perturbed_videos")

    env = gym.make(args.env)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Add-ons
    # 1. properly determine action_dim
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
        action_dim = env.actoin_space.n
        max_action = 1
        min_action = 0
    else:
        raise NotImplementedError

    # 2. determine state dimensions
    if isinstance(env.observation_space, spaces.Discrete):
        state_dim = 1
        max_state = env.observation_space.n - 1
    else:
        state_dim = env.observation_space.shape[0]
        max_state = np.inf
        
    # 3. check device
    if args.device not in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'auto']:
        raise NotImplementedError

    # 4. check hard or not
    if args.hard == 'False':
        hard = False
    else:
        hard = True
        
    # 5. check d4rl option and mixed option
    # determine data_path, log_path and save_path
    # get perturbed environment
    i = args.env.find('-')

    if args.env=='pilcocartpole-v0' or args.env=='lavapath-v0' or args.env=='bacreacher-v0':
        perturbed_env = f'{args.env[:i]}perturbed{args.env[i:]}'
    else:
        perturbed_env = f'{args.env[:i]}Perturbed{args.env[i:]}'
    #perturbed_env = f'{args.env[:i]}Perturbed{args.env[i:]}'  
    if args.d4rl == 'False' and args.mixed == 'False':
        if args.dtalrnt == 'True':
            load_path = f'FQI_{args.env}_dataeps{args.data_eps}_datapol{args.gendata_pol}_nsamples_{args.nsamples}_{args.comment}'
            save_path = f'FQI_{perturbed_env}_dataeps{args.data_eps}_datapol{args.gendata_pol}_nsamples_{args.nsamples}_{args.comment}'
        else:
            save_path = f'FQI_{perturbed_env}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
            load_path = f'FQI_{args.env}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
    elif args.d4rl == 'True' and args.mixed == 'False':  
        save_path = f'FQI_{perturbed_env}_d4rl'
        load_path = f'FQI_{args.env}_d4rl'
        if args.d4rl_expert == 'True':
            save_path += '_expert'
            load_path += '_expert'
        save_path += args.comment
        load_path += args.comment
    elif args.d4rl == 'False' and args.mixed == 'True':
        save_path = f'FQI_mixed_{perturbed_env}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
        load_path = f'FQI_mixed_{args.env}_dataeps{args.data_eps}_datapol{args.gendata_pol}{args.comment}'
    else:
        raise NotImplementedError
    if args.eval_episodes >20 :
        save_path=save_path+str(args.eval_episodes)

    paths = dict(load_path=load_path,
                    save_path=save_path) 
    print(paths)
    # broadcast
    i = args.env.find('-')
    #perturbed_env = f'{args.env[:i]}Perturbed{args.env[i:]}'
    print("=========================================================")
    print(f'===============Eval. FQI on {perturbed_env}==============')
    print(f'{args.env} attributes: max_action={max_action}')
    print(f'                       min_action={min_action}')
    print(f'                       action_dim={action_dim}')
    print(f'                       env action type is {env_action_type}')
    print(f'Eval. attributes:      using device: {args.device}')
    print(f'                       eval episodes: {args.eval_episodes}')
    print(f'FQI attributes:        beta_percentile={args.beta_percentile}')
    if args.d4rl == 'True':
        print(f'                       trained on d4rl')
    elif args.mixed == 'True':
        print(f'                       trained on data with eps={args.data_eps}')
        print(f'                       data collecting policy is WITH mixed')
    print(f'                       extra comment: {args.comment}')
    print("=========================================================")
    eval_FQI(state_dim, action_dim, max_state, min_action, max_action, hard,
                perturbed_env, args)
 
        