import numpy as np
from pyglet.window import key


class Policy(object):
    """
    Individual agent policy.
    """

    def __init__(self):
        pass

    def action(self, obs):
        raise NotImplementedError()


