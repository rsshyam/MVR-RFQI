"""
Network that takes in list of different sequence encoders and outputs values.

Ahtor: Ian Char
Date: January 6, 2023
"""
import torch

from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks.seq_encoders.seq_encoder_bundler import SeqEncoderBundler


class SeqEncoderQNet(PyTorchModule):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_depth: int,
        hidden_width: int,
        encoders: SeqEncoderBundler,
        track_encoder_grads: bool = True,
        append_current_observations: bool = False,
        layer_norm_encodings: bool = True,
        init_w: float = 1e-3,
    ):
        """Construct.

        Args:
            obs_dim: Observation dim.
            act_dim: The action dim.
            hidden_depth: The hidden depth.
            hidden_width: The hidden width.
            encoders: List of sequence encoders to be appended.
            track_encoder_grads: Whether to track gradients from the encoders.
            append_current_observations: Whether to append the current observation
                to the encodings.
            layer_norm_encodings: Whether to apply layer norm to the encodings.
        """
        super().__init__()
        self.encoders = encoders
        self.append_current_observations = append_current_observations
        self.track_encoder_grads = track_encoder_grads
        self.decoder = Mlp(
            input_size=(encoders.output_dim + append_current_observations * obs_dim
                        + act_dim),
            output_size=1,
            hidden_sizes=[hidden_width for _ in range(hidden_depth)],
        )
        if layer_norm_encodings:
            self.layer_norm = torch.nn.LayerNorm(self.encoders.output_dim)
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
        # TODO: Add previous reward sequence here.
        encoded = self.encoders(obs_seq, prev_act_seq, None, masks)
        if not self.track_encoder_grads:
            encoded = encoded.detach()
        if self.layer_norm is not None:
            encoded = self.layer_norm(encoded)
        if self.append_current_observations:
            encoded = torch.cat([obs_seq[:, -1], encoded], dim=1)
        encoded = torch.cat([encoded, act], dim=1)
        return self.decoder(encoded)
