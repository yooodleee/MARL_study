
import copy
import torch
import torch.nn.functional as F
from torch.optim import RMSprop, Adam
import numpy as np


from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer



class QLearner:

    def __init__(
            self,
            mac,
            scheme,
            logger,
            args,
    ):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if self.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            
            else:
                raise ValueError("Mixer {} not recognized.".format(args.mixer))
            
            self.params += list(self.mixer.parameters())
        
        if self.args.optimizer == "adam":
            self.optimizer = Adam(
                params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0)
            )
        else:
            self.optimizer = RMSprop(
                params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
            )
        

        # a little wasteful to deepcopy (e.g. duplicates act selector), but shoud work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1


    