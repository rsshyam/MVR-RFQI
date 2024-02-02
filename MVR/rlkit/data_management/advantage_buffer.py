"""
Buffer that also stores advantage estimates as described in PPO.

Author: Ian Char
Date: 4/7/2021
"""
import numpy as np
import torch

from rlkit.data_management.onpolicy_buffer import OnPolicyBuffer
from rlkit.torch.core import torch_ify


class AdvantageReplayBuffer(OnPolicyBuffer):
    """Buffer that also stores advantages."""

    def __init__(
        self,
        max_replay_buffer_size,
        env,
        value_function,
        tdlambda=0.95,
        discount=0.99,
        target_lookahead=10,
    ):
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observation_dim = env.observation_space.low.size
        self._action_dim = env.action_space.low.size
        self._value_function = value_function
        self._tdlambda = tdlambda
        self._discount = discount
        self._target_lookahead = target_lookahead
        self.clear_buffer()

    def clear_buffer(self):
        """Clear the replay buffers."""
        # Set up buffers.
        self._observations = np.zeros((self._max_replay_buffer_size,
                                       self._observation_dim))
        self._next_obs = np.zeros((self._max_replay_buffer_size,
                                   self._observation_dim))
        self._actions = np.zeros((self._max_replay_buffer_size,
                                  self._action_dim))
        self._rewards = np.zeros((self._max_replay_buffer_size, 1))
        self._advantages = np.zeros((self._max_replay_buffer_size, 1))
        self._targets = np.zeros((self._max_replay_buffer_size, 1))
        self._logpis = np.zeros((self._max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((self._max_replay_buffer_size, 1),
                                   dtype='uint8')
        self._top = 0
        self._size = 0

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            advantages=self._advantages[indices],
            targets=self._targets[indices],
            logpis=self._logpis[indices],
        )
        return batch

    def add_path(self, path):
        obs = path['observations']
        acts = path['actions']
        rews = path['rewards']
        nxts = path['next_observations']
        dones = path['terminals']
        pis = np.array([ai['logpi'] for ai in path['agent_infos']])
        path_len = len(obs)
        # Compute the advantages.
        with torch.no_grad():
            vals = self._value_function(torch_ify(obs)).cpu().numpy()
            nxt_vals = self._value_function(torch_ify(nxts)).cpu().numpy()
        deltas = rews + self._discount * nxt_vals * (1 - dones) - vals
        advantages = np.zeros(path_len)
        for idx, delta in enumerate(deltas.flatten()[::-1]):
            if idx == 0:
                advantages[-1] = delta
            else:
                advantages[path_len - 1 - idx] = (delta
                        + self._discount * self._tdlambda
                        * advantages[path_len - idx])
        targets = advantages + vals.flatten()
        for o, a, r, n, d, ad, pi, t in zip(obs, acts, rews, nxts, dones,
                                            advantages, pis, targets):
            self._observations[self._top] = o
            self._actions[self._top] = a
            self._rewards[self._top] = r
            self._terminals[self._top] = d
            self._next_obs[self._top] = n
            self._advantages[self._top] = ad
            self._targets[self._top] = t
            self._logpis[self._top] = pi

            self._advance()

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1
