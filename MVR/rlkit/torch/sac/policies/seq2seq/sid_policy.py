"""
Sequential policies from research.

Author: Ian Char
Date: December 11, 2022
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


class SIDPolicy(TorchStochasticSequencePolicy):
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
        obs_encoder_width: int,
        obs_encoder_depth: int,
        obs_encoding_size: int,
        act_encoder_width: int,
        act_encoder_depth: int,
        act_encoding_size: int,
        lookback_len: int,
        std=None,
        use_act_encoder: bool = False,
        attach_obs: bool = False,
        sum_over_terms: bool = False,
        detach_i_gradients: bool = False,
        init_w: float = 1e-3,
    ):
        """Constructor.

        Args:
            For each of the encoders provide width, depth, and the encoding produced.
            lookback_len: How long ago should we look back for the integral term.
            std: Fixed standard deviation if necessary.
        """
        super().__init__()
        self.obs_encoder = Mlp(
            input_size=obs_dim,
            output_size=obs_encoding_size,
            hidden_sizes=[obs_encoder_width for _ in range(obs_encoder_depth)],
        )
        self.lookback_len = lookback_len
        self.use_act_encoder = use_act_encoder
        self.attach_obs = attach_obs
        self.total_encode_dim = obs_encoding_size
        self.sum_over_terms = sum_over_terms
        self.detach_i_gradients = detach_i_gradients
        if use_act_encoder:
            self.act_encoder = Mlp(
                input_size=obs_dim,
                output_size=act_encoding_size,
                hidden_sizes=[act_encoder_width for _ in range(act_encoder_depth)],
            )
            self.total_encode_dim += act_encoder_depth
        else:
            self.act_encoder = None
        self.std = std
        in_dim = self.total_encode_dim * 3
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
        self.i_conv = torch.nn.Conv1d(
            in_channels=self.total_encode_dim,
            out_channels=self.total_encode_dim,
            kernel_size=lookback_len,
            groups=self.total_encode_dim,
            bias=False,
        )
        with torch.no_grad():
            self.i_conv.weight = torch.nn.Parameter(
                torch.ones_like(self.i_conv.weight) / float(self.lookback_len))
        self.i_conv.weight.requires_grad = False

    def forward(self, obs_seq, act_seq):
        """Forward should have shapes
            (batch_size, L, obs_dim) and (batch_size, L - 1, act_dim)
        """
        # Encode all of the sequences.
        stats = self.obs_encoder(obs_seq)
        if self.use_act_encoder:
            act_stats = self.act_encoder(act_seq)
            stats = torch.cat([stats, act_stats], dim=-1)
        # Pad the fron of the stats with lookback_len - 1 for integral term.
        padded_stats = torch.cat([
            ptu.zeros(stats.shape[0], self.lookback_len - 1, stats.shape[-1]),
            stats,
        ], dim=1)
        integral_stats = self.i_conv(torch.transpose(padded_stats, 1, 2))
        integral_stats = torch.transpose(integral_stats, 1, 2)
        diff_stats = (padded_stats[:, self.lookback_len - 1:]
                      - padded_stats[:, self.lookback_len - 2:-1])
        sid_out = torch.cat([
            stats,
            integral_stats,
            diff_stats,
        ], dim=-1)
        means = self.last_layer(sid_out)
        if self.std is None:
            log_std = self.last_fc_log_std(sid_out)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            stds = torch.exp(log_std)
        else:
            stds = ptu.ones(means.shape) * self.std
        return [TanhNormal(means[:, i], stds[:, i]) for i in range(means.shape[1])]


class SIDPolicyAdapter(TorchStochasticPolicy):

    def __init__(
        self,
        policy: SIDPolicy,
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
                                    self.policy.total_encode_dim))
        new_stats = self.policy.obs_encoder(obs)
        if self.policy.act_encoder is not None:
            if first_step:
                self.last_acts = ptu.zeros((len(obs),
                                            self.policy.act_encoder.input_size))
            act_stats = self.policy.act_encoder(self.last_acts)
            new_stats = torch.cat([new_stats, act_stats], dim=1)
        self.stats = torch.cat([self.stats[:, 1:], new_stats.unsqueeze(1)], dim=1)
        integral_stat = torch.mean(self.stats, dim=1)
        diff_stat = self.stats[:, -1] - self.stats[:, -2]
        curr_sid = torch.cat([self.stats[:, -1], integral_stat, diff_stat], dim=-1)
        mean = self.policy.last_layer(curr_sid)
        if self.policy.std is None:
            log_std = self.policy.last_fc_log_std(curr_sid)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.policy.std, ])).float().to(
                ptu.device)
        return TanhNormal(mean, std)
