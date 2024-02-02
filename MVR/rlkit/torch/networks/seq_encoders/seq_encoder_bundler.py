"""
Module that takes a bunch of sequence encodings and bundles them together into one.

Author: Ian Char

Date: January 6, 2023
"""
from typing import Sequence

import torch

from rlkit.torch.networks.seq_encoders.seq_encoder import SequenceEncoder


class SeqEncoderBundler(torch.nn.Module):

    def __init__(
        self,
        encoders: Sequence[SequenceEncoder],
    ):
        super().__init__()
        self.num_encoders = len(encoders)
        for enc_idx, enc in enumerate(encoders):
            setattr(self, f'encoder_{enc_idx}', enc)

    def forward(
        self,
        obs: torch.Tensor,
        acts: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Call all of the encoders and concat.

        Args:
            obs: Observation sequence of shape (batch_size, L, obs_dim).
            acts: Action sequence of shape (batch_size, L, act_dim).
            rewards: Reward sequence of shape (batch_size, L, 1).
            masks: Mask of real data vs padded 0s (batch_size, L, 1).

        Returns: Encoding of shape (batch_size, encode_dim)
        """
        encodings = torch.cat([
            getattr(self, f'encoder_{enc_idx}').encode(obs, acts, rewards, masks)
            for enc_idx in range(self.num_encoders)], dim=1)
        return encodings

    @property
    def output_dim(self) -> int:
        return sum([getattr(self, f'encoder_{i}').output_dim
                    for i in range(self.num_encoders)])

    @property
    def lookback(self) -> int:
        """The lookback needed to make the encoding."""
        return max([getattr(self, f'encoder_{i}').lookback
                    for i in range(self.num_encoders)])
