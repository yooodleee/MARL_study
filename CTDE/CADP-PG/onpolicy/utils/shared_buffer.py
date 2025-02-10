
import torch
import numpy as np
from onpolicy.utils.util import get_shape_from_obs_space
from onpolicy.utils.util import get_shape_from_act_space



def _flatten(T, N, x):

    return x.reshape(T * N, *x.shape[2:])



def _cast(x):

    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])




class SharedReplayBuffer(object):
    """
    Buffer to store training data.

    Params
    ----------------
        args: (argparse.Namespace)
            args containing relevant model, policy, and env info.
        num_agents: (int)
            num of agents in the env.
        obs_space: (gym.Space)
            obs space of agents.
        cent_obs_space: (gym.Space)
            centralized obs space of agents.
        act_space: (gym.Space)
            act space for agents.
    """

    def __init__(
            self,
            args,
            num_agents,
            obs_space,
            cent_obs_space,
            act_space):
        
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]
        
        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]
        

        self.share_obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *share_obs_shape
            ),
            dtype=np.float32
        )

        self.obs = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                *obs_shape,
            ),
            dtype=np.float32
        )

        self.rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32
        )

        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
                1,
            ),
            dtype=np.float32
        )

        self.returns = np.zeros_like(self.value_preds)


        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones(
                (
                    self.episode_length + 1,
                    self.n_rollout_threads,
                    num_agents,
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
                num_agents,
                act_shape,
            ),
            dtype=np.float32
        )

        self.action_log_probs = np.zeros(
            (
                self.episode_length,
                self.n_rollout_threads,
                num_agents,
                act_shape,
            ),
            dtype=np.float32
        )

        self.rewards = np.zeros(
            (
                self.episode_length,
                self.n_rollout_threads,
                num_agents,
                1,
            ),
            dtype=np.float32
        )

        self.masks = np.ones(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                num_agents,
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
            rnn_states_actor,
            rnn_states_critic,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            masks,
            bad_masks=None,
            active_masks=None,
            available_actions=None):
        
        """
        Insert data into the buffer.

        Params
        ----------------
            share_obs: (argparse.Namespace)
                args containing relevent model, policy, and env info.
            obs: (np.ndarray)
                local agent obs.
            rnn_states_actor: (np.ndarray)
                RNN states for actor network.
            rnn_states_critic: (np.ndarray)
                RNN states for critic network.
            actions: (np.ndarray)
                acts taken by agents.
            action_log_probs: (np.ndarray)
                log probs of acts taken by agents.
            value_preds: (np.ndarray)
                value func pred at each step.
            rewards: (np.ndarray)
                reward collected at each step.
            masks: (np.ndarray)
                denotes whether the env has terminated or not.
            bad_masks: (np.ndarray)
                act space for agents.
            active_masks: (np.ndarray)
                denotes whether an agent is active or dead in the env.
            available_actions: (np.ndarray)
                acts available to each agent. If None, all acts are available.
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
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
    

    