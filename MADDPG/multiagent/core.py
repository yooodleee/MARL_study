import numpy as np


# physical/external base state of all entites
class EntityState(object):

    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


