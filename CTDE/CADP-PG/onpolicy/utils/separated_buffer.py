
import torch
import numpy as np
from collections import defaultdict

from onpolicy.utils.util import check
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.utils.util import get_shape_from_act_space



def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 0, 2).reshape(-1, *x.shape[2:])



class SeparatedReplayBuffer(object):

    def __init__(
            self,
            args,
            obs_space,
            share_obs_space,
            act_space):
        
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.rnn_hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits


        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_space = get_shape_from_obs_space(share_obs_space)


        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]
        

        if type(share_obs_space[-1]) == list:
            share_obs_space = share_obs_space[1:]
        

        self.share_obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                *share_obs_space,
            ),
            dtype=np.float32
        )

        self.obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                *obs_shape,
            ),
            dtype=np.float32
        )


        self.rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.recurrent_N,
                self.rnn_hidden_size,
            ),
            dtype=np.float32
        )

        self.rnn_states_critic = np.zeros_like(self.rnn_states)


        self.value_preds = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                1,
            ),
            dtype=np.float32
        )

        self.returns = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                1,
            ),
            dtype=np.float32
        )


        if act_space.__class__.__name__ == "Discrete":
            self.available_actions = np.ones(
                (
                    self.episode_length + 1,
                    self.n_rollout_threads,
                    act_space.n,
                ),
                dtype=np.float32
            )
        else:
            self.available_actions = None
        

        act_shape = get_shape_from_act_space(act_space)


        self.actions = np.zeros(
            (
                self.episode_length,
                self.n_rollout_threads,
                act_shape,
            ),
            dtype=np.float32
        )

        self.action_log_probs = np.zeros(
            (
                self.episode_length,
                self.n_rollout_threads,
                act_shape,
            ),
            dtype=np.float32
        )

        self.rewards = np.zeros(
            (
                self.episode_length,
                self.n_rollout_threads,
                1,
            ),
            dtype=np.float32
        )


        self.masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                1,
            ),
            dtype=np.float32
        )
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0
    

    def insert(
            self,
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            masks,
            bad_masks=None,
            active_masks=None,
            available_actions=None):
        
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()


        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()
        
        self.step = (self.step + 1) % self.episode_length
    

    def chooseinsert(
            self,
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            masks,
            bad_masks=None,
            active_masks=None,
            available_actions=None):
        
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        if bad_masks is not None:
            self.bad_masks[self.step + 1] =bad_masks.copy()
        
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length
    

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()

        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()
    

    def chooseafter_update(self):
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
    

    def compute_returns(
            self,
            next_value,
            value_normalizer=None):
        
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0

                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] \
                                + self.gamma \
                                * value_normalizer.denormalize(self.value_preds[step + 1]) \
                                * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]

                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    
                    else:
                        delta = self.rewards[step] \
                                + self.gamma \
                                * self.value_preds[step + 1] \
                                * self.masks[step + 1] \
                                - self.value_preds[step]
                        
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]

                        self.returns[step] = gae + self.value_preds[step]

            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart:
                        self.returns[step] = (
                            self.returns[step + 1] \
                            * self.gamma \
                            * self.masks[step + 1] \
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) \
                        * value_normalizer.denormalize(self.value_preds[step])
                    
                    else:
                        self.returns[step] = (
                            self.returns[step + 1] \
                            * self.gamma \
                            * self.masks[step + 1] \
                            + self.rewards[step]
                        ) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) \
                        * self.value_preds[step]
        
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0

                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] \
                            + self.gamma \
                            * value_normalizer.denormalize(self.value_preds[step + 1]) \
                            * self.masks[step + 1] \
                            - value_normalizer.denormalize(self.value_preds[step])
                        
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
            
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] \
                                        * self.gamma \
                                        + self.masks[step + 1] \
                                        + self.rewards[step]
    

    