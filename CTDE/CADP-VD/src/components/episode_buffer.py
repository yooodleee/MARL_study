import torch
import numpy as np
from types import SimpleNamespace as SN




class EpisodeBatch:

    def __init__(
            self,
            scheme,
            groups,
            batch_size,
            max_seq_length,
            data=None,
            preprocess=None,
            device="cpu"):
        
        self.scheme = scheme
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)
    

    