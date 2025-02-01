"""
Scenario:
1 speaker, 2 listeners (one of which is an adversary).
Good agents rewarded for proximality to gaol, and distance from adversary
    to gaol. Adversary is rewarded for its distance to the goal.
"""

import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random


class CryptoAgent(Agent):

    def __init__(self):
        super(CryptoAgent, self).__init__()
        self.key = None
    

