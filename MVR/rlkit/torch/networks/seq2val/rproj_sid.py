"""
Network that forms encoding based on random projections and their 

Author: Ian Char
Date: December 12, 2022
"""
import torch

from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.core import PyTorchModule


class RprojSIDQNet(PyTorchModule):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_projections: int,
        lookback_len: int,
        decoder_width: int,
        decoder_depth: int,
        layer_norm: bool = True,
        sum_over_terms: bool = False,
        proj_init_w=1.0,
    ):
        """Constructor.

        Args:
            obs_dim: Size of the observation dim.
            act_dim: Size of the action dim.
            encode_size: Size of the statistic.
            lookback_len: The lookback to consider for the integral.
            encoder_width: Width of the hidden units in the encoder.
            encoder_depth: Number of hidden units in the encoder.
            decoder_width: Width of the hidden units in the decoder.
            decoder_depth: Number of hidden units in the decoder.
            encode_action_seq: Whether to encode past action sequence.
        """
        super().__init__()
        self.lookback_len = lookback_len
        self.sum_over_terms = sum_over_terms
        self.projections = torch.nn.Linear(obs_dim, num_projections, bias=False)
        self.projections.weight.data.uniform_(-proj_init_w, proj_init_w)
        self.projections.weight.requires_grad = False
        self.decoder = Mlp(
            input_size=num_projections * 3 + obs_dim + act_dim,
            output_size=1,
            hidden_sizes=[decoder_width for _ in range(decoder_depth)],
        )
        if layer_norm:
            self.layer_norm = torch.nn.LayerNorm(3 * num_projections)
        else:
            self.layer_norm = None

    def forward(self, obs_seq, prev_act_seq, act, masks=None, **kwargs):
        """Forward pass.

        Args:
            obs_seq: Observation sequence (batch_size, L, obs_dim)
            prev_act_seq: Previous action sequence (batch_size, L, act_dim)
            act: The current action (batch_size, act_dim)

        Returns: Value for last observation + action (batch_size, 1)
        """
        stats = self.projections(obs_seq)
        if masks is not None:
            stats *= masks
        if self.sum_over_terms:
            iterm = torch.sum(stats, dim=1)
        else:
            iterm = torch.mean(stats, dim=1)
        sid_out = torch.cat([
            stats[:, -1],
            iterm,
            stats[:, -1] - stats[:, -2],
        ], dim=-1)
        if self.layer_norm is not None:
            sid_out = self.layer_norm(sid_out)
        return self.decoder(torch.cat([
            obs_seq[:, -1],
            act,
            sid_out,
        ], dim=-1))
