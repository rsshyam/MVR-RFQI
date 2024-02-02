"""
Recurrent policies.

Author: Ian Char
Date: December 10, 2022
"""
import abc

import torch
import numpy as np

from rlkit.policies.base import SeqeunceExplorationPolicy
from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy
from rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)


class TorchStochasticSequencePolicy(
    DistributionGenerator,
    SeqeunceExplorationPolicy, metaclass=abc.ABCMeta
):
    def get_action(self, obs_np, prev_act_np=None):
        if prev_act_np is not None:
            prev_act_np = prev_act_np[None]
        actions, logprobs = self.get_actions(obs_np[None], prev_act_np)
        return actions[0, :], {'logpi': float(logprobs)}

    def get_actions(self, obs_np, prev_act_np=None):
        """Get actions.

        Args:
            obs_np: ndarray with shape (batch_size, L, obs_dim)
            prev_act_np: ndarray with shape (batch_size, L, act_dim)
                prev_act_np at index i of sequence is at time t-1.
                obs_np at index i of sequence is at time t.

        Returns:
            Actions and logprobs and ndarray with shape (batch_size, L, act_dim)
            and (batch_size, L, 1)
        """
        dists = self.get_dist_from_np(obs_np, prev_act_np)
        try:
            dist_outputs = [dist.sample_and_logprob() for dist in dists]
            actions, logprobs = [
                torch.cat([do[i].unsqueeze(1) for do in dist_outputs], dim=1)
                for i in range(2)
            ]
            logprobs = logprobs.unsqueeze(1)
        except NotImplementedError:
            actions = torch.cat([dist.sample().unsqueeze(1) for dist in dists], dim=1)
            logprobs = np.ones((actions.shape[0], actions.shape[1], 1))
        return (elem_or_tuple_to_numpy(actions),
                elem_or_tuple_to_numpy(logprobs))

    def get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(torch_ify(x) if x is not None else None for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist
