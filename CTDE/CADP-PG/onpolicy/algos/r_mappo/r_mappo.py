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
    

    