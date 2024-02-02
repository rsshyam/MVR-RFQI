"""
Network that does a convolution over computed statistics.

Author: Ian Char
Date: December 12, 2022
"""
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.core import PyTorchModule


class TemporalConvNet(PyTorchModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        encode_size: int,
        lookback_len: int,
        num_channels: int,
        encoder_width: int,
        encoder_depth: int,
        decoder_width: int,
        decoder_depth: int,
    ):
        """Constructor.

        Args:
            input_size: Size of the input.
            output_size: Size of the output.
            encode_size: Size of the statistic.
            lookback_len: The lookback to consider for the integral.
            num_channels: The number of convolutions to learn over.
            encoder_width: Width of the hidden units in the encoder.
            encoder_depth: Number of hidden units in the encoder.
            decoder_width: Width of the hidden units in the decoder.
            decoder_depth: Number of hidden units in the decoder.
        """
        super().__init__()
        self.lookback_len = lookback_len
        self.encoder = Mlp(
            input_size=input_size,
            output_size=encode_size,
            hidden_sizes=[encoder_width for _ in range(encoder_depth)],
        )
        self.decoder = Mlp(
            input_size=num_channels,
            output_size=output_size,
            hidden_sizes=[decoder_width for _ in range(decoder_depth)],
        )
        self.conv = torch.nn.Conv1d(
            in_channels=encode_size,
            out_channels=num_channels,
            kernel_size=lookback_len,
        )

    def forward(self, net_in, **kwargs):
        """Forward pass.

        Args:
            net_in: This should have shape (batch_size, L, input_dim)

        Returns: Output with shape (batch_size, L, output_dim)
        """
        stats = self.encoder(net_in)
        padded_stats = torch.cat([
            ptu.zeros(stats.shape[0], self.lookback_len - 1, stats.shape[-1]),
            stats,
        ], dim=1)
        conv_out = self.conv(torch.transpose(padded_stats, 1, 2))
        conv_out = torch.transpose(conv_out, 1, 2)
        return self.decoder(conv_out)


class FlattenTemporalConvNet(TemporalConvNet):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """
    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)
