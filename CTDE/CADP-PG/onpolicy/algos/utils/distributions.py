
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
    

    