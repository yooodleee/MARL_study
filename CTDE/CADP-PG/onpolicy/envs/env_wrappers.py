
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
    

    def __getstate__(self):
        import cloudpickle
        
        return cloudpickle.dumps(self.x)
    

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)



