
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop, Adam


from components.episode_buffer import EpisodeBatch
from modules.mixers.qplex import DMAQ_Q_attenMixer
from utils.th_utils import get_params_size



def entropy(x, dim=-1):
    max_entropy = np.log(x.shape[dim])
    x = (x + 1e-8) / torch.sum(x + 1e-8, dim, keepdim=True)

    return (-torch.log(x) * x).sum(dim) / max_entropy



