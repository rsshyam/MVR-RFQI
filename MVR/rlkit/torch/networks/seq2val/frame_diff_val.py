"""
Value function that uses frame stacking.

Author: Ian Char
Date: December 12, 2022
"""
import torch

from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.core import PyTorchModule


class FrameDiffQNet(PyTorchModule):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_depth: int,
        hidden_width: int,
        diff_coef: float = 1.0,
    ):
        """Constructor.

        Args:
            obs_dim: Size of the observation dim.
            act_dim: Size of the action dim.
            encode_size: Size of the statistic.
            hidden_depth: Depth of the network.
            hidden_width: Width of the network.
            lookback_len: The lookback to consider for the integral.
            encode_action_seq: Whether to encode past action sequence.
        """
        super().__init__()
        self.diff_coef = diff_coef
        in_dim = obs_dim * 2 + act_dim
        self.net = Mlp(
            input_size=in_dim,
            output_size=1,
            hidden_sizes=[hidden_width for _ in range(hidden_depth)],
        )

    def forward(self, obs_seq, prev_act_seq, act, **kwargs):
        """Forward pass.

        Args:
            obs_seq: Observation sequence (batch_size, L, obs_dim)
            prev_act_seq: Previous action sequence (batch_size, L, act_dim)
            act: The current action (batch_size, act_dim)

        Returns: Value for last observation + action (batch_size, 1)
        """
        net_in = torch.cat([
            obs_seq[:, -1],
            (obs_seq[:, -1] - obs_seq[:, -2]) * self.diff_coef,
            act,
        ], dim=-1)
        return self.net(net_in)
