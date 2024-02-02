import logging
from gym.envs.registration import register
from bax.envs.pendulum import PendulumEnv, pendulum_reward
from bax.envs.pilco_cartpole import CartPoleSwingUpEnv, pilco_cartpole_reward
from bax.envs.pilcocartpole_perturbed import CartPolePerturbedEnv
from bax.envs.pendulum_perturbed import PendulumPerturbedEnv
from bax.envs.goddard import GoddardEnv, goddard_reward
# from bax.util.envs.pets_cartpole import PETSCartpoleEnv, cartpole_reward
from bax.envs.acrobot import AcrobotEnv, acrobot_reward
from bax.envs.lava_path import LavaPathEnv, lava_path_reward


# register each environment we wanna use
register(
    id='bacpendulum-v0',
    entry_point=PendulumEnv,
    )

register(
    id='bacpendulumPerturbed-v0',
    entry_point=PendulumPerturbedEnv,
    )

# register(
#    id='hopper_v3',
#    entry_point=HopperEnv,
#    )

register(
    id='bacpendulum-tight-v0',
    entry_point=PendulumEnv,
    kwargs={'tight_start': True}
    )

register(
    id='bacpendulum-medium-v0',
    entry_point=PendulumEnv,
    kwargs={'medium_start': True}
    )

register(
    id='goddard-v0',
    entry_point=GoddardEnv,
    )
# register(
#     id='petscartpole-v0',
#     entry_point=PETSCartpoleEnv,
#     )
register(
    id='pilcocartpole-v0',
    entry_point=CartPoleSwingUpEnv,
    max_episode_steps=25
    )
register(
    id='pilcocartpoleperturbed-v0',
    entry_point=CartPolePerturbedEnv,
    max_episode_steps=25
    )
register(
    id='pilcocartpole-trig-v0',
    entry_point=CartPoleSwingUpEnv,
    kwargs={'use_trig': True},
    )
register(
    id='bacrobot-v0',
    entry_point=AcrobotEnv,
    )
register(
    id='lavapath-v0',
    entry_point=LavaPathEnv,
    )
reward_functions = {
        'bacpendulum-v0': pendulum_reward,
        'bacpendulum-tight-v0': pendulum_reward,
        'bacpendulum-medium-v0': pendulum_reward,
        'goddard-v0': goddard_reward,
        # 'petscartpole-v0': cartpole_reward,
        'pilcocartpole-v0': pilco_cartpole_reward,
        'pilcocartpole-trig-v0': pilco_cartpole_reward,
        'bacrobot-v0': acrobot_reward,
        'lavapath-v0': lava_path_reward,
        }


#register(
#    id='hopper_v3',
#    entry_point=HopperEnv,
#    )
# mujoco stuff
try:
    from bax.envs.swimmer import BACSwimmerEnv, swimmer_reward
    from bax.envs.reacher import BACReacherEnv, reacher_reward
    from bax.envs.reacher_perturbed import BACReacherPerturbedEnv
    register(
        id='bacswimmer-v0',
        entry_point=BACSwimmerEnv,
        )
    register(
        id='bacreacher-v0',
        entry_point=BACReacherEnv,
        )
    register(
        id='bacreacherperturbed-v0',
        entry_point=BACReacherPerturbedEnv,
        )
    register(
        id='bacreacher-tight-v0',
        entry_point=BACReacherEnv,
        kwargs={'tight': True},
        )
    reward_functions['bacswimmer-v0'] = swimmer_reward
    reward_functions['bacreacher-v0'] = reacher_reward
    reward_functions['bacreacher-tight-v0'] = reacher_reward
except:
    logging.info('mujoco not found, skipping those envs')
try:
    from bax.envs.beta_tracking_env import BetaTrackingGymEnv, beta_tracking_rew
    from bax.envs.tracking_env import TrackingGymEnv, tracking_rew
    register(
        id='betatracking-v0',
        entry_point=BetaTrackingGymEnv,
        )
    register(
        id='plasmatracking-v0',
        entry_point=TrackingGymEnv,
        )
    reward_functions['plasmatracking-v0'] = tracking_rew
    reward_functions['betatracking-v0'] = beta_tracking_rew
except:
    logging.info('fusion dependencies not found, skipping')
