import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.utils.util import get_grad_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algos.utils.util import check



class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.

    Params
    ---------
    args (argparse.Namespace)
        arguments containing relevant model, policy, and env info.
    policy (R_MAPPO_Policy)
        policy to update.
    device (torch.device)
        specifies the device to run on (cpu/gpu).
    """

    def __init__(
            self,
            args,
            policy,
            device=torch.device("cpu")):
        
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        assert (self._use_popart and self._use_valuenorm) \
            == False, \
            ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1).to(self.device)
        
        else:
            self.value_normalizer = None
        
        if getattr(self.args, 'use_CADP', False):
            self.use_cadp_loss = False
    

    def cal_value_loss(
            self,
            values,
            value_preds_batch,
            return_batch,
            active_masks_batch):
        
        """
        Calculate value function loss.

        Params
        ----------
            - values (torch.Tensor)
                value function predictions.
            - value_preds_batch (torch.Tensor)
                "old" value predictions from data batch (used for value clip loss).
            - return batch (torch.Tensor)
                reward to go returns.
            -active_masks_batch (torch.Tensor)
                denotes if agent is active or dead at a given timesep.

        Return
        ----------
            - value_loss (torch.Tensor)
                value function loss.
        """
        value_pred_clipped = \
            value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        

        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values
        

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)
        

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss_clipped = value_loss_original
        

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()
        

        return value_loss
    

    def ppo_update(
            self,
            sample,
            update_actor=True):
        
        """
        Update actor and critic networks.

        Params
        ---------
            - sample (Tuple)
                contains data batch with which to update networks.
            - update_actor (bool)
                whether to update actor network.

        
        Returns
        ------------
            - value_loss (torch.Tensor)
                value function loss.
            - critic_grad_norm (torch.Tensor)
                gradient norm from critic up9data.
            - policy_loss (torch.Tensor)
                actor(policy) loss value.
            - dist_entropy (torch.Tensor)
                action entropies.
            - actor_grad_norm (torch.Tensor)
                gradient norm from actor update.
            - imp_weights (torch.Tensor)
                importance sampling weights.
        """
        share_obs_batch, \
        obs_batch, \
        rnn_states_batch, \
        rnn_states_critic_batch, \
        actions_batch, \
        value_preds_batch, \
        return_batch, \
        masks_batch, \
        active_masks_batch, \
        old_action_log_probs_batch, \
        adv_trag, \
        available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_trag = check(adv_trag).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)


        # reshape to do in a single forward pass for all steps
        values, \
        action_log_progs, \
        dist_entropy = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # actor update
        imp_weights = torch.exp(action_log_progs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_trag
        surr2 = torch.clamp(
            imp_weights,
            1.0 - self.clip_param,
            1.0 + self.clip_param,
        ) * adv_trag


        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()
        
        policy_loss = policy_action_loss


        if getattr(self.args, 'use_CADP', False):
            if self.use_cadp_loss:

                att = self.policy.actor.att.dot
                eps = 1e-8
                eye = torch.eye(self.args.num_agents).unsqueeze(0).repeat(att.shape[0], 1, 1)
                eye = eye.view(-1, self.args.num_agents).to(**self.tpdv)
                att = att.view(-1, self.args.num_agents)

                cadp_loss = F.kl_div((att + eps).log(), eye, reduction='mean')
                policy_loss = policy_loss + 0.5 * cadp_loss
            

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()
            

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())
            
        self.policy.actor_optimizer.step()


        # critic update
        value_loss = self.cal_value_loss(
            values,
            value_preds_batch, 
            return_batch, 
            active_masks_batch,
        )

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()


        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())
            
        self.policy.critic_optimizer.step()


        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights
        

    def train(
            self,
            buffer,
            update_actor=True):
        
        """
        Performs a training update using minibatch GD.

        
        params
        ---------
            - buffer (SharedReplayBuffer)
                buffer containing training data.
            - update_actor (bool)
                whether to update actor network.


        Return
        -----------
            - train_info (dict)
                contains info regarding training update 
                    (e.g. loss, grad norms, etc).
        """

        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] \
                        - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan

        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(
                    advantages,
                    self.num_mini_batch,
                    self.data_chunk_length,
                )
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(
                    advantages,
                    self.num_mini_batch,
                )
            else:
                data_generator = buffer.feed_forward_generator(
                    advantages,
                    self.num_mini_batch,
                )
        
            for sample in data_generator:
                value_loss, \
                critic_grad_norm, \
                policy_loss, \
                dist_entropy, \
                actor_grad_norm, \
                imp_weights = self.ppo_update(sample, update_actor)


                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm.item()
                train_info['critic_grad_norm'] += critic_grad_norm.item()
                train_info['ratio'] += imp_weights.mean()
        

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        
        
        return train_info
    

    