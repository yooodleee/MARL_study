
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC


import torch
import numpy as np



class NMAC(BasicMAC):
    """
    This multi-agent controller shares params between agents.
    """

    def __init__(
            self,
            scheme,
            groups,
            args):
        
        super(NMAC, self).__init__(scheme, groups, args)
    

    