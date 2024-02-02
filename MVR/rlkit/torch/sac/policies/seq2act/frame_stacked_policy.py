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


class FrameStackPolicy(TorchStochasticSequencePolicy):
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
        hidden_width: int,
        hidden_depth: int,
        lookback_len: int,
        std=None,
        use_act_encoder: bool = False,
        init_w: float = 1e-3,
    ):
        """Constructor.

        Args:
            For each of the encoders provide width, depth, and the encoding produced.
            lookback_len: How long ago should we look back for the integral term.
            std: Fixed standard deviation if necessary.
        """
        super().__init__()
        self.lookback_len = lookback_len
        self.use_act_encoder = use_act_encoder
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        in_dim = obs_dim * lookback_len
        if use_act_encoder:
            in_dim += action_dim * lookback_len
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
        h = obs_seq[:, -self.lookback_len:].view(obs_seq.shape[0], -1)
        if self.use_act_encoder:
            h = torch.cat([
                h,
                act_seq[:, -self.lookback_len:].view(obs_seq.shape[0], -1),
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


class FrameStackPolicyAdapter(TorchStochasticPolicy):

    def __init__(
        self,
        policy: FrameStackPolicy,
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
            self.obs_hist = ptu.zeros((len(obs), self.policy.lookback_len,
                                       self.policy.obs_dim))
            self.act_hist = ptu.zeros((len(obs), self.policy.lookback_len,
                                       self.policy.action_dim))
        self.obs_hist = torch.cat([
            self.obs_hist[:, 1:],
            obs.view(obs.shape[0], 1, obs.shape[1]),
        ], dim=1)
        return self.policy.forward(self.obs_hist, self.act_hist)
