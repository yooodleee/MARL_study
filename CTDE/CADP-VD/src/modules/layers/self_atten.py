
import torch
import torch.nn as nn
import torch.nn.functional as F



class SelfAttention(nn.Module):

    def __init__(
            self,
            input_size,
            heads,
            embed_size
    ):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size

        self.tokeys = nn.Linear(self.input_size, self.emb_size * heads, bias=False)
        self.toqueries = nn.Linear(self.input_size, self.emb_size * heads, bias= False)
        self.tovalues = nn.Linear(self.input_size, self.emb_size * heads, bias=False)

    
    