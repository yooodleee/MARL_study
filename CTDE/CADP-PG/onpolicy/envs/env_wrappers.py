
"""
Modified from OpenAI Baselines code to work with multi-agent envs.
"""

import numpy as np
import torch

from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
from onpolicy.utils.util import tile_images



class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries
        to use pickle).
    """

    def __init__(self, x):
        self.x = x
    

    