
import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algos.utils.util import init, check
from onpolicy.algos.utils.cnn import CNNBase
from onpolicy.algos.utils.mlp import MLPBase
from onpolicy.algos.utils.rnn import RNNLayer
from onpolicy.algos.utils.act import ACTLayer
from onpolicy.algos.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space



class SelfAttention(nn.Module):

    def __init__(
            self,
            input_size,
            heads=4,
            embed_size=32):
        
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size

        self.tokeys = nn.Linear(
            self.input_size,
            self.emb_size * heads,
            bias=False
        )
        self.toqueries = nn.Linear(
            self.input_size,
            self.emb_size * heads,
            bias=False
        )
        self.tovalues = nn.Linear(
            self.input_size,
            self.emb_size * heads,
            bias=False
        )
    

    