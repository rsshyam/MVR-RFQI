"""
Policy that simply concatenates previous observations/actions together.

Author: Ian Char
Date: December 23, 2022
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


class FrameDiffPolicy(TorchStochasticSequencePolicy):
    """Policy that takes in current observation and the difference between the last
    observations.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_width: int,
        hidden_depth: int,
        diff_coef: float = 1.0,
        std=None,
        init_w: float = 1e-3,
    ):
        """Constructor.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.diff_coef = diff_coef
        in_dim = obs_dim * 2
        self.net = Mlp(
            input_size=in_dim,
            output_size=action_dim,
            init_w=init_w,
            hidden_sizes=[hidden_width for _ in range(hidden_depth)],
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = hidden_width if hidden_depth > 0 else in_dim
            self.last_fc_log_std = torch.nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs_seq, act_seq, masks=None):
        """Forward should have shapes

        Args:
            obs_seq: The observation sequence.
            act_seq: The previous action sequence.
        """
        # Do the input reshaping.
        h = torch.cat([
            obs_seq[:, -1],
            (obs_seq[:, -1] - obs_seq[:, -2]) * self.diff_coef,
        ], dim=-1)
        for i, fc in enumerate(self.net.fcs):
            h = self.net.hidden_activation(fc(h))
        mean = self.net.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return TanhNormal(mean, std)


class FrameDiffPolicyAdapter(TorchStochasticPolicy):

    def __init__(
        self,
        policy: FrameDiffPolicy,
        keep_track_of_grads: bool = False,
    ):
        super().__init__()
        self.policy = policy
        self.keep_track_of_grads = keep_track_of_grads
        self.reset()

    def reset(self):
        self.obs_hist = None
        self.act_hist = None
        self.last_acts = None

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
        self.last_acts = actions.detach()
        return (elem_or_tuple_to_numpy(actions),
                elem_or_tuple_to_numpy(logprobs))

    def forward(self, obs):
        """Do a forward pass. It is assumed that the amount of observations
           will remain the same batch size throughout the episode.
        """
        first_step = self.obs_hist is None
        if first_step:
            self.obs_hist = ptu.zeros((len(obs), 2, self.policy.obs_dim))
            self.act_hist = ptu.zeros((len(obs), 2, self.policy.action_dim))
        self.obs_hist = torch.cat([
            self.obs_hist[:, 1:],
            obs.view(obs.shape[0], 1, obs.shape[1]),
        ], dim=1)
        return self.policy.forward(self.obs_hist, self.act_hist)
