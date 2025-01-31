import numpy as np


# physical/external base state of all entites
class EntityState(object):

    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):

    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None


# action of the agent
class Action(object):

    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


