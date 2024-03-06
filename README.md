This code is associated to the paper DISTRIBUTIONALLY ROBUST MODEL-BASED REINFORCEMENT LEARNING WITH LARGE STATE SPACES https://arxiv.org/pdf/2309.02236.pdf 




The codebase is forked from RFQI https://github.com/zaiyan-x/RFQI (authored by Zaiyan Xu) and BARL https://github.com/fusion-ml/trajectory-information-rl/tree/main, https://github.com/fusion-ml/bac-baselines/tree/master (both authored by Viraj Mehta) codebase from previous works. As stated in RFQI codebase, the code requires mujoco and D4RL pre-installed

The perturbed environments on which we test the algorithm have to be defined as provided instructions in https://github.com/zaiyan-x/RFQI 

Clone conda environments rqitry2 and rqicombwthrlkit2 from the environment files rqitry2.yml and rqicombwthrlkit2.yml

(Or)

**MVR Environment**
Create a python 3.9.16 conda environment, and install swig package in the environment by `conda install -c conda-forge swig` and `pip install mvr_requirements.txt` to run the MVR related scripts.

**RFQI Environment**
Create conda env with python 3.9.16

```
pip install gym==0.23.1
pip install pygame
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install tensorboard==2.10.0
pip install imageio==2.9.0
pip install stable-baselines3==1.1.0
```
Install mujoco by following
https://pytorch.org/rl/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html

`pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl`


We separate the data generation and training process into 2 different environments as they were not compatible with each other. 

**Part-1 (Data-generation from various environments from MVR Folder)**
All environment definitions and initiations has been done in bax.envs folder. The perturbed environment definitions similar to RFQI codebase has been initialized in bax.envs folder.

Activate rqicombwthrlkit2 environment for this part
 
(the code also uses rlkit2 codebase https://github.com/IanChar/rlkit2/tree/main . Especially, the rlkit folder has been downloaded into the main directory for the datagnr_old_baselines.py code to use the same. The SAC implementation of this is used in the reacher environment)

The SAC in pendulum corresponds to from stable-baselines

Run the scripts for each environment (true and learnt) to generate the data

The scripts that run for true model automatically store the policies in .pt files. Store the policies in model folder and then the baseline eval scripts for baseline evaluation.

**Part-2 (Offline Data Transfer)**

Offline data corresponding to each experiment/environement is stored in the experiments folder under $experiment_name_data$/time/offline_data

Copy all these files to the RFQI/offline_data folder so that train_rfqi.py can use it. 

**Part-3 (RFQI/FQI Training)**

For all experiments we use rqitry2 environment.

As the data been generated from slightly different definitions of the environments (except pendulum), both their environment definitions and their perturbed environement definitions mentioned as  RFQI/perturbed_env/pilco_cartpole.py, pilcocartpole_perturbed.py, bacreacher.py, bacreacher_perturbed.py, pendulum.py, pendulum_perturbed.py) from Part-1 need to be defined again similar to the process mentioned in https://github.com/zaiyan-x/RFQI inside the conda environment’s gym/envs file. In particular add pendulum,cartpole environments in gym/classic_control and in gym/classic_control/__init__.py add

```
from gym.envs.classic_control.pendulum import PendulumEnv 
from gym.envs.classic_control.pendulum_perturbed import PendulumPerturbedEnv 
from gym.envs.classic_control.pilco_cartpole import CartPoleSwingUpEnv
from gym.envs.classic_control.pilcocartpole_perturbed import CartPolePerturbedEnv

Add reacher environments in gym/mujoco and in gym/mujoco/__init__.py add
from gym.envs.mujoco.bacreacher import BACReacherEnv
from gym.envs.mujoco.bacreacher import BACReacherPerturbedEnv
```

Then, under gym/envs/__init__.py add
```
register(
    id='pilcocartpole-v0',
    entry_point="gym.envs.classic_control.pilco_cartpole:CartPoleSwingUpEnv",
    max_episode_steps=25,
    reward_threshold=195.0,
    )

register(
    id="pilcocartpoleperturbed-v0",
    entry_point="gym.envs.classic_control.pilcocartpole_perturbed:CartPolePerturbedEnv",
    max_episode_steps=25,
    reward_threshold=195.0,
)

register(
    id="Pendulum-v1",
    entry_point="gym.envs.classic_control:PendulumEnv",
    max_episode_steps=200,
)

register(
    id="PendulumPerturbed-v1",
    entry_point="gym.envs.classic_control.pendulum_perturbed:PendulumPerturbedEnv",
    max_episode_steps=200,
)
register(
        id='bacreacher-v0',
        entry_point="gym.envs.mujoco:BACReacherEnv",
        max_episode_steps=50,
)
register(
        id='bacreacherperturbed-v0',
        entry_point="gym.envs.mujoco.bacreacher_perturbed:BACReacherPerturbedEnv",
        max_episode_steps=50,
)
```

Note that we have defined the reacher from Part-1 as ‘bac-reacher’ to differentiate it from the pre-defined reacher environment.

Due to this change in environment names, make sure to change the first part of the offline data file names mentioning the environment name ‘bac-reacher-v0’, ‘Pendulum-v1’ and ‘pilco_cartpole-v0’ so that the train_rfqi and eval_rfqi files recognise the data corresponding to the environments.

Easiest way is to use the run_seq_env_script.sh with mentioning all the parameters such as environment and training hyperparameters. 

The training and evaluation scripts for each environment using run_seq_env_script.sh are provided in the scripts folder for RFQI/FQI. Changing from rfqi to fqi just requires a change in type as mentioned in the scripts

Further we use the wandb website to log the data. So it has to be installed and logged in properly.

Results of the evaluation will be stored in the perturbed_results folder, using which one can create the plots from the paper.

**Reference:**
```
@article{ramesh2023distributionally,
  title={Distributionally robust model-based reinforcement learning with large state spaces},
  author={Ramesh, Shyam Sundhar and Sessa, Pier Giuseppe and Hu, Yifan and Krause, Andreas and Bogunovic, Ilija},
  journal={arXiv preprint arXiv:2309.02236},
  year={2023}
}
```
