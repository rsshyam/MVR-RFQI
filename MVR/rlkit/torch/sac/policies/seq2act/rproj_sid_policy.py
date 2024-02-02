"""
Sequential policy that takes integral, derivative of random projections.

Author: Ian Char
Date: December 19, 2022
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


class RprojSIDPolicy(TorchStochasticSequencePolicy):
    """Policy that comes up with statistics for each of the states. The action is
    then a linear function of
        * The last current statistic.
        * The integral over the last few statistics.
        * The difference of the last statistics.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_projections: int,
        lookback_len: int,
        std=None,
        attach_obs: bool = False,
        layer_norm: bool = True,
        sum_over_terms: bool = False,
        init_w: float = 1e-3,
        proj_init_w: float = 1e-3,
    ):
        """Constructor.

        Args:
            For each of the encoders provide width, depth, and the encoding produced.
            lookback_len: How long ago should we look back for the integral term.
            std: Fixed standard deviation if necessary.
        """
        super().__init__()
        self.projections = torch.nn.Linear(obs_dim, num_projections, bias=False)
        self.projections.weight.data.uniform_(-proj_init_w, proj_init_w)
        self.projections.weight.requires_grad = False
        self.lookback_len = lookback_len
        self.attach_obs = attach_obs
        self.sum_over_terms = sum_over_terms
        self.std = std
        self.num_projections = num_projections
        in_dim = num_projections * 3
        if attach_obs:
            in_dim += obs_dim
        self.last_layer = torch.nn.Linear(in_dim, action_dim)
        if std is None:
            self.last_fc_log_std = torch.nn.Linear(in_dim, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
        if layer_norm:
            self.layer_norm = torch.nn.LayerNorm(3 * num_projections)
        else:
            self.layer_norm = None

    def forward(self, obs_seq, act_seq, masks=None):
        """Forward should have shapes
            (batch_size, L, obs_dim), (batch_size, L, act_dim), and (batch_size, L, 1)
        """
        # Encode all of the sequences.
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
        ], dim=1)
        if self.layer_norm is not None:
            sid_out = self.layer_norm(sid_out)
        if self.attach_obs:
            sid_out = torch.cat([obs_seq[:, -1], sid_out], dim=-1)
        means = self.last_layer(sid_out)
        if self.std is None:
            log_std = self.last_fc_log_std(sid_out)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            stds = torch.exp(log_std)
        else:
            stds = ptu.ones(means.shape) * self.std
        return TanhNormal(means, stds)


class RprojSIDPolicyAdapter(TorchStochasticPolicy):

    def __init__(
        self,
        policy: RprojSIDPolicy,
        keep_track_of_grads: bool = False,
    ):
        super().__init__()
        self.policy = policy
        self.keep_track_of_grads = keep_track_of_grads
        self.reset()

    def reset(self):
        self.stats = None
        self.insert_pt = 0
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
        first_step = self.stats is None
        if first_step:
            self.stats = ptu.zeros((len(obs), self.policy.lookback_len,
                                    self.policy.num_projections))
        new_stats = self.policy.projections(obs)
        self.stats = torch.cat([self.stats[:, 1:], new_stats.unsqueeze(1)], dim=1)
        if self.policy.sum_over_terms:
            iterm = torch.sum(self.stats, dim=1)
        else:
            iterm = torch.mean(self.stats, dim=1)
        diff_stat = self.stats[:, -1] - self.stats[:, -2]
        curr_sid = torch.cat([self.stats[:, -1], iterm, diff_stat], dim=-1)
        if self.policy.layer_norm is not None:
            curr_sid = self.policy.layer_norm(curr_sid)
        if self.policy.attach_obs:
            curr_sid = torch.cat([obs, curr_sid], dim=-1)
        mean = self.policy.last_layer(curr_sid)
        if self.policy.std is None:
            log_std = self.policy.last_fc_log_std(curr_sid)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.policy.std, ])).float().to(
                ptu.device)
        return TanhNormal(mean, std)
