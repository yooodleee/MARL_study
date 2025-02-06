
import torch.nn as nn
from .util import init


"""
CNN Modules and utils.
"""


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)



