
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
    

    @abstractmethod
    def reset(self):
        """
        Reset all the envs and return an array of obs, or a dict 
            of observation arrays. If step_async is still work,
            that work will be cancelled and step_wait() should 
            not be called until step_async() is invoked again.
        """
        pass


    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the envs to start talking a step with the given acts.
        Call step_wait() to get the results of the step.

        Should not call this if a step_async run is already pending.
        """
        pass


    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        
        Returns: (obs, rews, dones, infos)
        ---------------------------------------
            obs:
                an array of obs, or a dict of arrays of obs.
            rews:
                an array of rewards.
            dones:
                an array of "episode done" booleans.
            infos:
                a sequence of info objects.
        """
        pass


    def close_extras(self):
        """
        Clean up the extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass


    def close(self):
        if self.closed:
            return
        
        if self.viewer is not None:
            self.viewer.close()
        
        self.close_extras()
        self.closed = True


    def step(self, actions):
        """
        Step the envs synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)

        return self.step_wait()
    

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)

        if mode == 'human':
            self.get_viewer().imshow(bigimg)

            return self.get_viewer().isopen
        
        elif mode == 'rgb_array':
            return bigimg
        
        else:
            raise NotImplementedError
    

    def get_images(self):
        """
        Return RGB images from each env.
        """
        raise NotImplementedError
    

    @property
    def unwrapped(self):
        if isinstance(self, ValueError):
            return self.venv.unwrapped
        
        else:
            return self
    

    