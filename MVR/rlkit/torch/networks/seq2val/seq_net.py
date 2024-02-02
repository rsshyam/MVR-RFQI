"""
Network that forms value from the lastest observation and action only.

Author: Ian Char
Date: December 12, 2022
"""
import torch

from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.core import PyTorchModule


class SeqQNet(PyTorchModule):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_depth: int,
        hidden_width: int,
    ):
        """Constructor.

        Args:
            obs_dim: Size of the observation dim.
            act_dim: Size of the action dim.
            hidden_depth: Depth of the network.
            hidden_width: Width of the network.
        """
        super().__init__()
        in_dim = obs_dim + act_dim
        self.net = Mlp(
            input_size=in_dim,
            output_size=1,
            hidden_sizes=[hidden_width for _ in range(hidden_depth)],
        )

    def forward(self, obs_seq, prev_act_seq, act, masks=None, **kwargs):
        """Forward pass.

        Args:
            obs_seq: Observation sequence (batch_size, L, obs_dim)
            prev_act_seq: Previous action sequence (batch_size, L, act_dim)
            act: The current action (batch_size, act_dim)

        Returns: Value for last observation + action (batch_size, 1)
        """
        return self.net(torch.cat([obs_seq[:, -1], act], dim=1))
