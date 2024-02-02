"""
Sequential policy that takes integral, derivative of statistics.

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
from rlkit.torch.networks.mlp import Mlp


class IndvSIDPolicy(TorchStochasticSequencePolicy):
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
        layer_norm: bool = True,
        init_w: float = 1e-3,
    ):
        """Constructor.

        Args:
            For each of the encoders provide width, depth, and the encoding produced.
            lookback_len: How long ago should we look back for the integral term.
            std: Fixed standard deviation if necessary.
        """
        super().__init__()
        self.s_encoder, self.i_encoder, self.d_encoder = [
            Mlp(
                input_size=obs_dim,
                output_size=obs_encoding_size,
                hidden_sizes=[obs_encoder_width for _ in range(obs_encoder_depth)],
            ) for _ in range(3)]
        self.lookback_len = lookback_len
        self.use_act_encoder = use_act_encoder
        self.attach_obs = attach_obs
        self.total_encode_dim = obs_encoding_size
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
        if layer_norm:
            self.layer_norm = torch.nn.LayerNorm(3 * self.total_encode_dim)
        else:
            self.layer_norm = None

    def forward(self, obs_seq, act_seq, masks=None):
        """Forward should have shapes
            (batch_size, L, obs_dim) and (batch_size, L, act_dim)
        """
        # Encode all of the sequences.
        s_terms = self.s_encoder(obs_seq[:, -1])
        i_terms = self.i_encoder(obs_seq)
        d_terms = self.d_encoder(obs_seq[:, -2:])
        if masks is not None:
            d_terms *= masks[:, -2:]
            i_terms *= masks
        sid_out = torch.cat([
            s_terms,
            torch.mean(i_terms, dim=1),
            d_terms[:, -1] - d_terms[:, -2],
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


class IndvSIDPolicyAdapter(TorchStochasticPolicy):

    def __init__(
        self,
        policy: IndvSIDPolicy,
        keep_track_of_grads: bool = False,
    ):
        super().__init__()
        self.policy = policy
        self.keep_track_of_grads = keep_track_of_grads
        self.reset()

    def reset(self):
        self.s_stats, self.i_stats, self.d_stats = None, None, None
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
        first_step = self.s_stats is None
        if first_step:
            self.s_stats, self.i_stats, self.d_stats = [
                ptu.zeros((len(obs), self.policy.lookback_len,
                          self.policy.total_encode_dim)) for _ in range(3)]
        s_stat = self.policy.s_encoder(obs)
        i_stat = self.policy.i_encoder(obs)
        d_stat = self.policy.d_encoder(obs)
        if self.policy.act_encoder is not None:
            raise NotImplementedError('TODO')
        self.s_stats = torch.cat([self.s_stats[:, 1:], s_stat.unsqueeze(1)], dim=1)
        self.i_stats = torch.cat([self.i_stats[:, 1:], i_stat.unsqueeze(1)], dim=1)
        self.d_stats = torch.cat([self.d_stats[:, 1:], d_stat.unsqueeze(1)], dim=1)
        iterm = torch.mean(self.i_stats, dim=1)
        diff_stat = self.d_stats[:, -1] - self.d_stats[:, -2]
        curr_sid = torch.cat([self.s_stats[:, -1], iterm, diff_stat], dim=-1)
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
