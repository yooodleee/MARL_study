import numpy as np


class BaseScenario(object):
    """
    Defines scenario upon which the world is built.
    """

    def make_world(self):
        """
        Create elements of the world.
        """
        raise NotImplementedError()
    
    def reset_world(self, world):
        """
        Create initial conditions of the world.
        """
        raise NotImplementedError()