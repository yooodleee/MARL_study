
import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner import Runner



def _t2n(x):

    return x.detach().cpu().numpy()



class SMACRunner(Runner):
    """
    Runner class to perform training, eval. and data collection for SMAC.
    See parent class for details.
    """

    def __init__(self, config):
        super(SMACRunner, self).__init__(config)
    

    