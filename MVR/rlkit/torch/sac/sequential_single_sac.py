"""
Version of SAC that takes in sequential data and then the policy/value network
only gives estimates for the last thing in the sequence.

Author: Ian Char
Date: December 17, 2022
"""
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from rlkit.core.loss import LossStatistics

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.logging import add_prefix
from rlkit.torch.sac.sac import SACTrainer, SACLosses


class SequentialSingleSACTrainer(SACTrainer):

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[SACLosses, LossStatistics]:
        """Compute the loss.

        Args:
            batch: For this version we expect the batch to have.
                observations: (batch_len, L, obs_dim)
                next_observations: (batch_len, L, obs_dim)
                actions: (batch_len, L, act_dim)
                prev_actions: (batch_len, L, act_dim)
                terminals: (batch_len, L, 1)
                masks: (batch_len, L, 1)
                rewards: (batch_len, L, 1)
        """
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        prev_acts = batch['prev_actions']
        masks = batch['masks']
        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs, prev_acts, masks)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha
                           * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1
        q_new_actions = torch.min(
            self.qf1(obs, prev_acts, new_obs_actions, masks),
            self.qf2(obs, prev_acts, new_obs_actions, masks),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, prev_acts, actions[:, -1], masks)
        q2_pred = self.qf2(obs, prev_acts, actions[:, -1], masks)
        next_dist = self.policy(next_obs, actions, masks)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_values = torch.min(
            self.target_qf1(next_obs, actions, new_next_actions, masks),
            self.target_qf2(next_obs, actions, new_next_actions, masks),
        ) - alpha * new_log_pi
        q_target = (self.reward_scale * rewards
                    + (1. - terminals) * self.discount * target_q_values)
        q_target = q_target.detach()
        if self.max_value is not None:
            q_target = torch.clamp(q_target, max=self.max_value)
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()
        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        )
        return loss, eval_statistics
