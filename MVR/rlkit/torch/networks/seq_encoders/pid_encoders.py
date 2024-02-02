"""
Encoders inspired by the PID controller.

Author: Ian Char
Date: Jan 4, 2022
"""
import torch

from rlkit.torch.networks.seq_encoders.basic_seq_encoder import BasicSeqEncoder


class PEncoder(BasicSeqEncoder):
    """
    Encoder inspired by the P term of PID. Simply makes an encoding of the current
    time step.
    """
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
        if len(net_in.shape) == 3:
            net_in = net_in[:, 0]
        return self.encoder(net_in)


class DEncoder(BasicSeqEncoder):
    """
    Encoder inspired by the D term  of PID. Takes the difference between two computed
    encodings at specified time steps apart.
    """

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
        net_out = self.encoder(net_in[:, [0, -1]]) * masks[:, [0, -1]]
        return net_out[:, -1] - net_out[:, 0]


class IEncoder(BasicSeqEncoder):
    """
    Encoder inspired by the I term  of PID. Takes the sum/mean of a statistic over
    time.
    """

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
        return (self.encoder(net_in) * masks).mean(dim=1)


class ExpIEncoder(BasicSeqEncoder):
    """
    Offshoot of the I encoder which does exponentially weighted averaging.
    """

    def __init__(
        self,
        alpha: float,
        **kwargs,
    ):
        """Constructor.

        Args:
            alpha: The smoothing parameter.
        """
        super().__init__(**kwargs)
        self.register_buffer('smoother', alpha * torch.Tensor([
            (1 - alpha) ** t
            for t in range(self.lookback - 1, -1, -1)
        ]).reshape(1, -1, 1))

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
        return (self.encoder(net_in) * masks * self.smoother).sum(dim=1)


class HardcodedPEncoder(PEncoder):
    """
    Encoder that hardcodes the difference between the variables.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert kwargs['obs_dim'] == 3
        assert self.encoding_size == 1
        with torch.no_grad():
            self.encoder.last_fc.weight = torch.nn.Parameter(
                torch.Tensor([[-1.0, 1.0, 0.0]]), requires_grad=False)
            self.encoder.last_fc.bias = torch.nn.Parameter(
                torch.Tensor([0.0]), requires_grad=False)


class HardcodedDEncoder(DEncoder):
    """
    Encoder that hardcodes the difference between the variables.
    """
    def __init__(self, dt: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        assert kwargs['obs_dim'] == 3
        assert self.encoding_size == 1
        with torch.no_grad():
            self.encoder.last_fc.weight = torch.nn.Parameter(
                torch.Tensor([[-1/dt, 1/dt, 0.0]]), requires_grad=False)
            self.encoder.last_fc.bias = torch.nn.Parameter(
                torch.Tensor([0.0]), requires_grad=False)


class HardcodedIEncoder(IEncoder):
    """
    Encoder that hardcodes the I term.
    """
    def __init__(self, dt: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        assert kwargs['obs_dim'] == 3
        assert self.encoding_size == 1
        with torch.no_grad():
            self.encoder.last_fc.weight = torch.nn.Parameter(
                torch.Tensor([[-dt, dt, 0.0]]), requires_grad=False)
            self.encoder.last_fc.bias = torch.nn.Parameter(
                torch.Tensor([0.0]), requires_grad=False)
