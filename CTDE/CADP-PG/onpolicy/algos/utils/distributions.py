
import torch
import torch.nn as nn
from .util import init


"""
Modify standard PyTorch distributions so they to make compatible
    with this codebase.
"""


# 
# Standardize distribution interfaces
#



# Categorical
class FixedCategorical(torch.distributions.Categorical):

    def sample(self):
        return super().sample().unsqueeze(-1)
    

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )
    
    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)



# Normal
class FixedNormal(torch.distributions.Normal):

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)
    

    def entropy(self):
        return self.entropy().sum(-1)
    

    def mode(self):
        return self.mean



# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):

    def log_probs(self, actions):
        return super\
            .log_prob(actions)\
            .view(actions.size(0), -1)\
            .sum(-1)\
            .unsqueeze(-1)
    

    def entropy(self):
        return super().entropy().sum(-1)
    

    def mode(self):
        return torch.gt(self.probs, 0.5).float()



