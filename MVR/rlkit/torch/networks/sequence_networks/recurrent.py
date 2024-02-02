"""
Recurrent neural nets.

Author: Ian Char
Date: December 12, 2022
"""
import torch

from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.core import PyTorchModule


class RecurrentNet(PyTorchModule):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        encode_size: int,
        rnn_hidden_size: int,
        encoder_width: int,
        encoder_depth: int,
        decoder_width: int,
        decoder_depth: int,
        layer_norm=False,
        rnn_num_layers: int = 1,
        rnn_type: str = 'gru'
    ):
        """Constructor.

        Args:
            input_size: Size of the input.
            output_size: Size of the output.
            encode_size: Size of the encoding to produce.
            rnn_hidden_size: The size of the hidden unit.
            encoder_width: Width of the hidden units in the encoder.
            encoder_depth: Number of hidden units in the encoder.
            decoder_width: Width of the hidden units in the decoder.
            decoder_depth: Number of hidden units in the decoder.
            layer_norm: Whether to apply a layer norm after the encoding.
            rnn_num_layers: The number of layers in the recurrent unit.
            rnn_type: The type of the recurrent unit.
        """
        super().__init__()
        self.encoder = Mlp(
            input_size=input_size,
            output_size=encode_size,
            hidden_sizes=[encoder_width for _ in range(encoder_depth)],
        )
        self.decoder = Mlp(
            input_size=encode_size + rnn_hidden_size,
            output_size=output_size,
            hidden_sizes=[decoder_width for _ in range(decoder_depth)],
        )
        if rnn_type.lower() == 'gru':
            rnn_class = torch.nn.GRU
        elif rnn_type.lower() == 'lstm':
            rnn_class = torch.nn.LSTM
        else:
            raise ValueError(f'Cannot recognize RNN type {rnn_type}')
        self.memory_unit = rnn_class(encode_size, rnn_hidden_size,
                                     num_layers=rnn_num_layers,
                                     batch_first=True)
        self.use_layer_norm = layer_norm
        if layer_norm:
            self._layer_norm = torch.nn.LayerNorm(encode_size)

    def forward(self, net_in, **kwargs):
        """Forward pass.

        Args:
            net_in: This should have shape (batch_size, L, input_dim)

        Returns: Output with shape (batch_size, L, output_dim)
        """
        encoding = self.encoder(net_in)
        if self.use_layer_norm:
            encoding = self._layer_norm(encoding)
        mem_out = self.memory_unit(encoding)[0]
        output = self.decoder(torch.cat([encoding, mem_out], dim=-1))
        return output


class FlattenRecurrentNet(RecurrentNet):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """
    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)
