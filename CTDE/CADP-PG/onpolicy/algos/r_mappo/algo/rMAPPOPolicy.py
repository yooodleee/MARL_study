
import torch

from onpolicy.algos.r_mappo.algo.r_actor_critic import R_Actor, R_Critic
from onpolicy.algos.r_mappo.algo.r_actor_critic_cadp import R_Actor as R_Actor_CADP
from onpolicy.algos.r_mappo.algo.r_actor_critic_cadp import R_Critic as R_Critic_CADP
from onpolicy.utils.util import update_linear_schedule, get_params_size


class R_MAPPOPolicy:
    """
    MAPPO Policy class.
    Wraps actor and critic networks to compute actions and values function predictions.

    
    Params
    --------
        args: (argparse.Namespace)
            arguments containing relevant model and policy info.
        obs_space: (gym.Space)
            observation space.
        action_obs_space: (gym.Space)
            value function input space (centralized input for MAPPO, decentralized 
                for IPPO).
        action_space: (gym.Space)
            action space.
        device: (torch.device)
            specifies the device to run on (cpu/gpu).
    """

    def __init__(
            self,
            args,
            obs_space,
            cent_obs_space,
            act_space,
            device=torch.device("cpu")):
        
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space


        if getattr(args, 'use_CADP', False):
            self.actor = R_Actor_CADP(
                args,
                self.obs_space,
                self.act_space,
                self.device,
            )
            self.critic = R_Critic_CADP(
                args,
                self.share_obs_space,
                self.device,
            )
        else:
            self.actor = R_Actor(
                args,
                self.obs_space,
                self.act_space,
                self.device,
            )
            self.critic = R_Critic(
                args,
                self.share_obs_space,
                self.device,
            )
        
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        params_size = get_params_size(params)
        print(("params_size: {}".format(params_size)))

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    
    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.

        
        Params
        ---------
            episode: (int)
                current training episode.
            episodes: (int)
                total number of training episodes.
        """
        update_linear_schedule(
            self.actor_optimizer,
            episode,
            episodes,
            self.lr,
        )
        update_linear_schedule(
            self.critic_optimizer,
            episode,
            episodes,
            self.critic_lr,
        )


    def get_actions(
            self,
            cent_obs,
            obs,
            rnn_states_actor,
            rnn_states_critic,
            masks,
            available_actions=None,
            deterministic=False):
        
        """
        Compute actions and value function predictions for the given inputs.

        
        Params
        -----------
            cent_obs: (np.ndarray)
                centralized input to the critic.
            obs: (np.ndarray)
                local agent inputs to the actor.
            rnn_states_actor: (np.ndarray)
                if actor is RNN, RNN states for actor.
            rnn_states_critic: (np.ndarray)
                if critic is RNN, RNN states for critic.
            masks: (np.ndarray)
                denotes points at which RNN states should be reset.
            availble_actions: (np.ndarray)
                denotes which actions are available to agent (if None,
                all actions available).
            deterministic: (bool)
                whether the action should be mode of distribution or should
                be sampled.


        Returns
        -----------
            values: (torch.Tensor)
                value function predictions.
            actions: (torch.Tensor)
                actions to take.
            action_log_progs: (torch.Tensor)
                log probabilities of chosen actions.
            rnn_states_actor: (torch.Tensor)
                updated actor network RNN states.
            rnn_states_critic: (torch.Tensor)
                updated critic network RNN states.
        """
        actions, \
        action_log_probs, \
        rnn_states_actor = self.actor(
            obs,
            rnn_states_actor,
            masks,
            available_actions,
            deterministic,
        )

        values, rnn_states_critic = self.critic(
            cent_obs,
            rnn_states_critic,
            masks,
        )

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
    

    