"""
Network that forms encoding based on statistics, integral, and derivative at each step.
Each uses a different network for encoding.

Author: Ian Char
Date: December 12, 2022
"""
import torch

from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.core import PyTorchModule


class IndvSIDQNet(PyTorchModule):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        encode_size: int,
        lookback_len: int,
        encoder_width: int,
        encoder_depth: int,
        decoder_width: int,
        decoder_depth: int,
        encode_action_seq: bool = False,
        layer_norm: bool = True,
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
        self.encode_action_seq = encode_action_seq
        input_size = obs_dim
        if encode_action_seq:
            input_size += act_dim
        self.s_encoder, self.i_encoder, self.d_encoder = [
            Mlp(
                input_size=input_size,
                output_size=encode_size,
                hidden_sizes=[encoder_width for _ in range(encoder_depth)],
            ) for _ in range(3)]
        self.decoder = Mlp(
            input_size=encode_size * 3 + obs_dim + act_dim,
            output_size=1,
            hidden_sizes=[decoder_width for _ in range(decoder_depth)],
        )
        if layer_norm:
            self.layer_norm = torch.nn.LayerNorm(3 * encode_size)
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
        if self.encode_action_seq:
            net_in = torch.cat([obs_seq, prev_act_seq], dim=-1)
        else:
            net_in = obs_seq
        s_terms = self.s_encoder(net_in[:, -1])
        i_terms = self.i_encoder(net_in)
        d_terms = self.d_encoder(net_in[:, -2:])
        if masks is not None:
            d_terms *= masks[:, -2:]
            i_terms *= masks
        sid_out = torch.cat([
            s_terms,
            torch.mean(i_terms, dim=1),
            d_terms[:, -1] - d_terms[:, -2],
        ], dim=-1)
        if self.layer_norm is not None:
            sid_out = self.layer_norm(sid_out)
        return self.decoder(torch.cat([
            obs_seq[:, -1],
            act,
            sid_out,
        ], dim=-1))
