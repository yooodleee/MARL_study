
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop, Adam


from components.episode_buffer import EpisodeBatch
from modules.mixers.qplex import DMAQ_QattenMixer
from utils.th_utils import get_params_size



def entropy(x, dim=-1):
    max_entropy = np.log(x.shape[dim])
    x = (x + 1e-8) / torch.sum(x + 1e-8, dim, keepdim=True)

    return (-torch.log(x) * x).sum(dim) / max_entropy



class DMAQ_qattenLearner:

    def __init__(
            self,
            mac,
            scheme,
            logger,
            args
    ):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "dmaq_qatten":
                self.mixer = DMAQ_QattenMixer
            else:
                raise ValueError("Mixer {} not recognized.".format(args.mixer))

            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
        
        if self.args.optimizer == "adam":
            self.optimizer = Adam(params=self.params, lr=args.lr)
        else:
            self.optimizer = RMSprop(
                params=self.params, lr=args.lr, alpha=args.optimi_alpha, eps=args.optim_eps
            )
        

        # a little wasteful to deepcopy (e.g. duplicates act selector), but shoule work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_actions = self.args.n_actions
    

    