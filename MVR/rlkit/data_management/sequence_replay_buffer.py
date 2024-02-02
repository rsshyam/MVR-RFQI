"""
A replay buffer that keeps some window of history.

Author: Ian Char
Date: December 10, 2022
"""
from collections import OrderedDict
from typing import Dict

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np


class SequenceReplayBuffer(ReplayBuffer):
    def __init__(
            self,
            max_replay_buffer_size: int,
            env,
            max_path_length: int,
            batch_window_size: int,
    ):
        """
        Constructor.

        Args:
            max_replay_buffer_size: The maximum size of the replay buffer. One data
                point refers to one sequence of the data.
            env: The environment that experience is being collected from.
            max_path_length: The maximum path length to store.
            batch_window_size: The size of window that are in a batch.
        """
        self._observation_dim = get_dim(env.observation_space)
        self._action_dim = get_dim(env.action_space)
        self._max_replay_buffer_size = int(max_replay_buffer_size)
        self._window_size = int(batch_window_size)
        self._max_replay_buffer_size = int(max_replay_buffer_size)
        self._max_path_length = int(max_path_length)
        self._max_data_points = int(max_replay_buffer_size * max_path_length)
        self.clear_buffer()

    def clear_buffer(self):
        """Clear all of the buffers."""
        # Initialize datastructures to be 3D tensors now that there is history.
        # We pad the end of each buffer with 0s since this will make the code much
        # easier since we don't have to worry as much about valid starts for subseqs.
        # Also pad the beginning of the action buffer with 1 columns of 0s so that
        # we can access previous actions.
        self._observations = np.zeros(
            (self._max_replay_buffer_size,
             self._max_path_length + self._window_size,
             self._observation_dim))
        self._actions = np.zeros(
            (self._max_replay_buffer_size,
             self._max_path_length + self._window_size,
             self._action_dim))
        self._rewards = np.zeros(
            (self._max_replay_buffer_size,
             self._max_path_length + self._window_size - 1, 1))
        self._terminals = np.zeros(
            (self._max_replay_buffer_size,
             self._max_path_length + self._window_size - 1, 1), dtype='uint8')
        self._masks = np.zeros(
            (self._max_replay_buffer_size,
             self._max_path_length + self._window_size - 1, 1), dtype='uint8')
        # Initialize data structures to keep track of path lengths, top of buffer, etc.
        self._valid_starts = np.zeros((self._max_data_points,  2), dtype='uint32')
        self._pathlens = np.zeros(self._max_path_length, dtype='uint16')
        self._buffer_top = 0
        self._buffer_size = 0
        self._valid_top = 0
        self._valid_bottom = 0
        self._valid_size = 0

    def add_path(self, path: Dict[str, np.ndarray]):
        """
        Add a path to the replay buffer.

        Args:
            path: The path collected as a dictionary of ndarrays.
        """
        pathlen = len(path['actions'])
        self._observations[self._buffer_top, :pathlen] =\
            path['observations']
        self._observations[self._buffer_top, pathlen] = path['next_observations'][-1]
        self._actions[self._buffer_top, 1:pathlen + 1] = path['actions']
        self._rewards[self._buffer_top, :pathlen] = path['rewards']
        self._terminals[self._buffer_top, :pathlen] = path['terminals']
        self._masks[self._buffer_top, :pathlen] = 1
        self._masks[self._buffer_top, pathlen:] = 0
        # Update the valid idxs.
        to_the_end = np.min([self._max_data_points - self._valid_top, pathlen])
        if to_the_end > 0:
            self._valid_starts[self._valid_top:self._valid_top + to_the_end] =\
                np.concatenate([s.reshape(-1, 1) for s in [
                                np.ones(to_the_end) * self._buffer_top,
                                np.arange(to_the_end)]
                                ], axis=1)
        if to_the_end < pathlen:
            additional_amt = pathlen - to_the_end
            self._valid_starts[:additional_amt] = np.concatenate([
                s.reshape(-1, 1) for s in [
                    np.ones(additional_amt) * self._buffer_top,
                    np.arange(to_the_end, pathlen),
                ]
            ], axis=1)
        # Update the size tracker and the tracking pointers..
        self._advance(pathlen)

    def random_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Get a random batch of data.

        Args:
            batch_size: number of sequences to grab.

        Returns: Dictionary of observation, actions, masks, etc.
        """
        vidxs = (np.random.randint(self._valid_size, size=batch_size)
                 + self._valid_bottom) % self._max_data_points
        seq_starts = self._valid_starts[vidxs]
        batch = {}
        for key, buffer in (
                ('observations', self._observations),
                ('actions', self._actions),
                ('rewards', self._rewards),
                ('next_observations', self._observations),
                ('prev_actions', self._actions),
                ('terminals', self._terminals),
                ('masks', self._masks)):
            # Note that the actions are offset by 1 when they are loaded in.
            offset = int(key == 'next_observations' or key == 'actions')
            batch[key] = buffer[seq_starts[:, 0].reshape(-1, 1),
                                np.array([seq_starts[:, 1] + i + offset
                                          for i in range(self._window_size)]).T]
        return batch

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    def _advance(self, pathlen):
        """Update the buffer after adding the path.

        Args:
            pathlen: The length of the path just added.
        """
        # If we just overwrote a shot change the bottom of the valid indices.
        if self._buffer_size >= self._max_replay_buffer_size:
            self._valid_bottom += self._pathlens[self._buffer_top]
            self._valid_size -= self._pathlens[self._buffer_top]
        # Update the path lengths and valid index informatoin.
        self._pathlens[self._buffer_top] = pathlen
        self._valid_top = (self._valid_top + pathlen) % self._max_data_points
        self._valid_size += pathlen
        # Update the top of the buffer
        self._buffer_top = (self._buffer_top + 1) % self._max_replay_buffer_size
        if self._buffer_size < self._max_replay_buffer_size:
            self._buffer_size += 1

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        """
        Add a transition tuple.
        """
        raise NotImplementedError('This buffer does not support adding single samples')

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        return self._valid_size

    def get_diagnostics(self):
        return OrderedDict([
            ('num_paths', self._buffer_size),
            ('num_samples', self._valid_size),
        ])


if __name__ == '__main__':
    """Some testing"""
    import gym
    env = gym.make('Pendulum-v1')
    buffer = SequenceReplayBuffer(3, env, max_path_length=5, batch_window_size=5)
    # Fill the buffer with different sized paths.
    pathlens = [1 for _ in range(15)]
    pnum = 0
    for pl in pathlens:
        path = {k: [] for k in ['observations', 'actions', 'rewards', 'terminals']}
        path['observations'].append(env.reset())
        for _ in range(pl):
            act = env.action_space.sample()
            nxt, rew, term, _ = env.step(act)
            path['observations'].append(nxt)
            path['rewards'].append(rew)
            path['actions'].append(act)
            path['terminals'].append(term)
        path['next_observations'] = np.array(path['observations'][1:])
        path['observations'] = np.array(path['observations'][:-1])
        for k in ('actions', 'rewards', 'terminals'):
            path[k] = np.array(path[k])
        path['rewards'] = path['rewards'].reshape(-1, 1)
        path['terminals'] = path['terminals'].reshape(-1, 1)
        buffer.add_path(path)
        pnum += 1
    print('=' * 10, f'Path Number {pnum}', '=' * 10)
    print(buffer._observations[:, :, 0])
    print(buffer._valid_bottom, buffer._valid_top, buffer._valid_size)
    print(buffer._valid_starts)
    print(buffer._valid_starts)
    batch = buffer.random_batch(2)
    for k in ('observations', 'next_observations'):
        print(k, batch[k][:, :, 0])
    print(buffer._actions[:, :, 0])
    for k in ('prev_actions', 'actions'):
        print(k, batch[k][:, :, 0])
    print(batch['masks'])
    # Do another path.
    pathlens = [5, 5]
    for pl in pathlens:
        path = {k: [] for k in ['observations', 'actions', 'rewards', 'terminals']}
        path['observations'].append(env.reset())
        for _ in range(pl):
            act = env.action_space.sample()
            nxt, rew, term, _ = env.step(act)
            path['observations'].append(nxt)
            path['rewards'].append(rew)
            path['actions'].append(act)
            path['terminals'].append(term)
        path['next_observations'] = np.array(path['observations'][1:])
        path['observations'] = np.array(path['observations'][:-1])
        for k in ('actions', 'rewards', 'terminals'):
            path[k] = np.array(path[k])
        path['rewards'] = path['rewards'].reshape(-1, 1)
        path['terminals'] = path['terminals'].reshape(-1, 1)
        buffer.add_path(path)
        pnum += 1
        print('=' * 10, f'Path Number {pnum}', '=' * 10)
        print(buffer._observations[:, :, 0])
        print(buffer._valid_bottom, buffer._valid_top, buffer._valid_size)
        print(buffer._valid_starts)
