
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

    
    