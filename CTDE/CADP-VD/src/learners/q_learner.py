
import copy
import torch
from torch.optim import RMSprop


from components.episode_buffer import EpisodeBatch
from modules.miexers.vdn import VDNMixer
from modules.mixers.qmix import QMixer




class QLearner:

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
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            
            elif self.mixer == "qmix":
                self.mixer = QMixer(args)
            
            else:
                raise ValueError("Mixer {} not recongnised.".format(args.mixer))

            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
        
        self.optimizer = RMSprop(
            params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
        )

        # a little wasteful to deepcopy (e.g. duplicates act selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
    

    