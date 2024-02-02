"""
Encoder that creates the mean/sum between two statistics.

Author: Ian Char
Date: Jan 4, 2022
"""
import abc

from rlkit.torch.networks.seq_encoders.seq_encoder import SequenceEncoder
from rlkit.torch.networks.mlp import Mlp


class BasicSeqEncoder(SequenceEncoder, metaclass=abc.ABCMeta):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        target_input: str,
        lookback_len: int,
        encoding_size: int,
        net_width: int,
        net_depth: int,

    ):
        super().__init__(
            target_input=target_input,
            lookback_len=lookback_len,
        )
        if self.target_input == 'observations':
            in_dim = obs_dim
        elif self.target_input == 'actions':
            in_dim = act_dim
        else:
            in_dim = 1
        self.encoding_size = encoding_size
        self.encoder = Mlp(
            hidden_sizes=[net_width for _ in range(net_depth)],
            output_size=self.encoding_size,
            input_size=in_dim,
        )

    @property
    def output_dim(self) -> int:
        """The output dimension of the encoding."""
        return self.encoding_size
