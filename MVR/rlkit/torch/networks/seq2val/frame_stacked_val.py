"""
Value function that uses frame stacking.

Author: Ian Char
Date: December 12, 2022
"""
import torch

from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.core import PyTorchModule


class FrameStackedQNet(PyTorchModule):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_depth: int,
        hidden_width: int,
        lookback_len: int,
        encode_action_seq: bool = False,
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
        self.lookback_len = lookback_len
        self.encode_action_seq = encode_action_seq
        in_dim = obs_dim * lookback_len + act_dim
        if encode_action_seq:
            in_dim += act_dim * lookback_len
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
        net_in = obs_seq[:, -self.lookback_len:].view(obs_seq.shape[0], -1)
        if self.encode_action_seq:
            net_in = torch.cat([
                net_in,
                prev_act_seq[:, -self.lookback_len].view(obs_seq.shape[0], -1),
            ], dim=-1)
        net_in = torch.cat([
            net_in,
            act,
        ], dim=-1)
        return self.net(net_in)
