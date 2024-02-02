"""
A replay buffer that keeps some window of history, but it is assumed that
only the last point will be used for gradient updates.

Author: Ian Char
Date: December 17, 2022
"""
from collections import OrderedDict
from typing import Dict

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.envs.env_utils import get_dim
import numpy as np


class SequenceSingleReplayBuffer(ReplayBuffer):
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
        if max_replay_buffer_size >= 2 ** 32:
            raise ValueError('Cannot support buffer size greater than 2^32')
        if max_path_length >= 2 ** 16:
            raise ValueError('Cannot support path length longer than 2^16')
        self._observation_dim = get_dim(env.observation_space)
        self._action_dim = get_dim(env.action_space)
        self._max_replay_buffer_size = int(max_replay_buffer_size)
        self._window_size = int(batch_window_size)
        self._max_replay_buffer_size = int(max_replay_buffer_size)
        self._max_path_length = int(max_path_length)
        self._max_data_points = int(max_replay_buffer_size * max_path_length)
        self._pad_size = self._window_size - 1
        self.clear_buffer()

    def clear_buffer(self):
        """Clear all of the buffers."""
        # Initialize datastructures to be 3D tensors now that there is history.
        # We pad the end of each beginning with 0s since this will make the code much
        # easier since we don't have to worry as much about valid ends for subseqs.
        # Also pad the beginning of the action buffer with additional 1 column so
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
        self._valid_ends = np.zeros((self._max_data_points,  2), dtype='uint32')
        self._pathlens = np.zeros(self._max_replay_buffer_size, dtype='uint16')
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
        endpt = pathlen + self._pad_size
        self._observations[self._buffer_top, self._pad_size:endpt] =\
            path['observations']
        self._observations[self._buffer_top, endpt] = path['next_observations'][-1]
        self._actions[self._buffer_top, self._pad_size + 1:endpt + 1] = path['actions']
        self._rewards[self._buffer_top, self._pad_size:endpt] = path['rewards']
        self._terminals[self._buffer_top, self._pad_size:endpt] = path['terminals']
        self._masks[self._buffer_top, self._pad_size:endpt] = 1
        # Update the valid idxs.
        to_the_end = np.min([self._max_data_points - self._valid_top, pathlen])
        if to_the_end > 0:
            self._valid_ends[self._valid_top:self._valid_top + to_the_end] =\
                np.concatenate([s.reshape(-1, 1) for s in [
                                np.ones(to_the_end) * self._buffer_top,
                                np.arange(self._pad_size, self._pad_size + to_the_end)]
                                ], axis=1)
        if to_the_end < pathlen:
            additional_amt = pathlen - to_the_end
            self._valid_ends[:additional_amt] = np.concatenate([
                s.reshape(-1, 1) for s in [
                    np.ones(additional_amt) * self._buffer_top,
                    np.arange(self._pad_size + to_the_end, self._pad_size + pathlen),
                ]
            ], axis=1)
        # Update the size tracker and the tracking pointers..
        self._advance(pathlen)

    def random_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Get a random batch of data.

        Args:
            batch_size: number of sequences to grab.

        Returns: Dictionary of information for the update.
            observations: This is a history w shape (batch_size, L, obs_dim)
            actions: This is the history of actions (batch_size, L, act_dim)
            rewards: This is the rewards at the last point (batch_size, 1)
            next_observation: This is a history of nexts (batch_size, L, obs_dim)
            prev_actions: History of previous actions (batch_size, L, act_dim)
            terminals: Whether last time step is terminals (batch_size, 1)
            masks: 
        """
        vidxs = (np.random.randint(self._valid_size, size=batch_size)
                 + self._valid_bottom) % self._max_data_points
        seq_ends = self._valid_ends[vidxs]
        batch = {}
        for key, buffer in (
                ('observations', self._observations),
                ('actions', self._actions),
                ('next_observations', self._observations),
                ('prev_actions', self._actions),
                ('masks', self._masks)):
            # Note that the actions are offset by 1 when they are loaded in.
            offset = int(key == 'next_observations' or key == 'actions')
            batch[key] =\
                buffer[seq_ends[:, 0].reshape(-1, 1),
                       np.array([seq_ends[:, 1] - i + offset
                                 for i in range(self._window_size - 1, -1, -1)]).T]
        for key, buffer in (
                ('rewards', self._rewards),
                ('terminals', self._terminals)):
            batch[key] = buffer[seq_ends[:, 0], seq_ends[:, 1]]
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
    buffer = SequenceSingleReplayBuffer(3, env, max_path_length=5, batch_window_size=5)
    # Fill the buffer with different sized paths.
    pathlens = [5 for _ in range(3)]
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
    print(buffer._actions[:, :, 0])
    print(buffer._rewards)
    print(buffer._valid_bottom, buffer._valid_top, buffer._valid_size)
    print(buffer._valid_ends)
    batch = buffer.random_batch(2)
    for k in ('observations', 'next_observations'):
        print(k, batch[k][:, :, 0])
    print(buffer._actions[:, :, 0])
    for k in ('prev_actions', 'actions'):
        print(k, batch[k][:, :, 0])
    print('rewards', batch['rewards'])
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
        print(buffer._valid_ends)
