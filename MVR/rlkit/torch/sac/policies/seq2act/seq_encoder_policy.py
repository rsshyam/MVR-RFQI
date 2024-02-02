"""
Policy that takes in list of different sequence encoders.

Author: Ian Char
Date: January 6, 2023
"""
import numpy as np
import torch

from rlkit.torch.core import elem_or_tuple_to_numpy
from rlkit.torch.distributions import TanhNormal
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sac.policies.base import TorchStochasticPolicy
from rlkit.torch.sac.policies.gaussian_policy import LOG_SIG_MAX, LOG_SIG_MIN
from rlkit.torch.sac.policies.sequence_policies import (
    TorchStochasticSequencePolicy,
)
from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.networks.seq_encoders.seq_encoder_bundler import SeqEncoderBundler


class SeqEncoderPolicy(TorchStochasticSequencePolicy):
    """Policy that uses sequence encoders."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_depth: int,
        hidden_width: int,
        encoders: SeqEncoderBundler,
        std=None,
        track_encoder_grads: bool = True,
        append_current_observations: bool = False,
        layer_norm_encodings: bool = True,
        init_w: float = 1e-3,
    ):
        """Constructor.

        Args:
            hidden_depth: The hidden depth for the mean and standard deviation.
            hidden_width: The hidden width for the mean and standard deviation.
            encoders: List of sequence encoders to be appended.
            std: Fixed standard deviation if necessary.
            track_encoder_grads: Whether to track gradients from the encoders.
            append_current_observations: Whether to append the current observation
                to the encodings.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoders = encoders
        self.track_encoder_grads = track_encoder_grads
        self.append_current_observations = append_current_observations
        self.mean_decoder = Mlp(
            input_size=(encoders.output_dim + append_current_observations * obs_dim),
            output_size=action_dim,
            hidden_sizes=[hidden_width for _ in range(hidden_depth)],
            last_fc_use_uniform=False,
        )
        self.std = std
        if std is None:
            self.std_decoder = Mlp(
                input_size=(encoders.output_dim
                            + append_current_observations * obs_dim),
                output_size=action_dim,
                hidden_sizes=[hidden_width for _ in range(hidden_depth)],
                last_has_bias=True,
                init_w=init_w,
            )
        if layer_norm_encodings:
            self.layer_norm = torch.nn.LayerNorm(self.encoders.output_dim)
        else:
            self.layer_norm = None

    def forward(self, obs_seq, act_seq, masks=None):
        """Forward should have shapes
            (batch_size, L, obs_dim), (batch_size, L, act_dim), and (batch_size, L, 1)
        """
        # TODO: Add previous reward sequence here.
        encoding = self.encoders(obs_seq, act_seq, None, masks)
        if not self.track_encoder_grads:
            encoding = encoding.detach()
        if self.layer_norm is not None:
            encoding = self.layer_norm(encoding)
        if self.append_current_observations:
            encoding = torch.cat([obs_seq[:, -1], encoding], dim=1)
        means = self.mean_decoder(encoding)
        if self.std is None:
            log_std = self.std_decoder(encoding)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            stds = torch.exp(log_std)
        else:
            stds = ptu.ones(means.shape) * self.std
        return TanhNormal(means, stds)


class SeqEncoderPolicyAdapter(TorchStochasticPolicy):

    def __init__(
        self,
        policy: SeqEncoderPolicy,
        keep_track_of_grads: bool = False,
    ):
        super().__init__()
        self.policy = policy
        self.keep_track_of_grads = keep_track_of_grads
        self.reset()

    def reset(self):
        self.obs_hist = None
        self.act_hist = None
        self.mask_hist = None
        # TODO: In the future need to figure out how to do rewards. Will need to
        #       receive this information from some external source.
        # self.rew_hist = None

    def get_actions(self, obs_np, ):
        if self.keep_track_of_grads:
            dist = self._get_dist_from_np(obs_np)
        else:
            with torch.no_grad():
                dist = self._get_dist_from_np(obs_np)
        try:
            actions, logprobs = dist.sample_and_logprob()
        except NotImplementedError:
            actions = dist.sample()
            logprobs = np.array(actions.shape[0])
        self.act_hist = torch.cat([self.act_hist[:, 1:],
                                   actions.detach().unsqueeze(1)], dim=1)
        return (elem_or_tuple_to_numpy(actions),
                elem_or_tuple_to_numpy(logprobs))

    def forward(self, obs):
        """Do a forward pass. It is assumed that the amount of observations
           will remain the same batch size throughout the episode.
        """
        first_step = self.obs_hist is None
        if first_step:
            self.obs_hist = ptu.zeros((len(obs), self.policy.encoders.lookback,
                                       self.policy.obs_dim))
            self.act_hist = ptu.zeros((len(obs), self.policy.encoders.lookback,
                                       self.policy.action_dim))
            self.mask_hist = ptu.zeros((len(obs), self.policy.encoders.lookback, 1))
        self.obs_hist = torch.cat([self.obs_hist[:, 1:], obs.unsqueeze(1)], dim=1)
        self.mask_hist = torch.cat([self.mask_hist[:, 1:],
                                    ptu.ones((len(self.mask_hist), 1, 1))], dim=1)
        return self.policy(self.obs_hist, self.act_hist, self.mask_hist)
