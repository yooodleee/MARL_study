
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
    

    