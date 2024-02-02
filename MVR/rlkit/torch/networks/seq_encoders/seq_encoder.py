"""
Abstract class for torch module that takes in some sequence of data and produces
an encoding.

Author: Ian Char
Date: Jan 4, 2022
"""
import abc

import torch


class SequenceEncoder(torch.nn.Module, metaclass=abc.ABCMeta):

    def __init__(
        self,
        target_input: str,
        lookback_len: int,
    ):
        """Constructor.

        Args:
            target_input: The input to look at, either observations, actions,
                or rewards.
            lookback_len: The number of steps backwards to consider.
        """
        super().__init__()
        assert target_input in ('observations', 'actions', 'rewards')
        self.target_input = target_input
        self.lookback_len = lookback_len

    def encode(
        self,
        obs: torch.Tensor,
        acts: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Make an encoding for the trajectory.

        Args:
            obs: Observation sequence of shape (batch_size, L, obs_dim).
            acts: Action sequence of shape (batch_size, L, act_dim).
            rewards: Reward sequence of shape (batch_size, L, 1).
            masks: Mask of real data vs padded 0s (batch_size, L, 1).

        Returns: Encoding of shape (batch_size, encode_dim)
        """
        if self.target_input == 'observations':
            encoder_input = obs
        elif self.target_input == 'actions':
            encoder_input = acts
        elif self.target_input == 'rewards':
            encoder_input = rewards
        else:
            raise ValueError('Target input type problem...')
        return self._make_encoding(
            encoder_input[:, -self.lookback_len:],
            masks[:, -self.lookback_len:],
        )

    @property
    def lookback(self) -> int:
        """The lookback needed to make the encoding."""
        return self.lookback_len

    @abc.abstractmethod
    def _make_encoding(
        self,
        net_in: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Make encoding.

        Args:
            net_in: Shape (batch_size, lookback_len, D)
            masks: Shape (batch_size, lookback_len, 1)

        Returns: Shape (batch_size, encode_dim)
        """

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """The output dimension of the encoding."""
