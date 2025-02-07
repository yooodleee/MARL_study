
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



class ShareVecEnv(ABC):
    """
    An obstract asynchronous, vectorized env. Used to batch data from multiple
        copies of an env, so that each obs becomes an batch of obs, and expected
        act is a batch of act to be applied per-env.
    """

    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    
    def __init__(
            self,
            num_envs,
            observation_space,
            share_observation_space,
            action_space):
        
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space
    

    