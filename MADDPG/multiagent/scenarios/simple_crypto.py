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
    

class Scenario(BaseScenario):

    def make_world(self):
        world = World()

        # set any world properties first
        num_agents = 3
        num_adversaries = 1
        num_landmarks = 2
        world.dim_c = 4

        # add agents
        world.agents = [CryptoAgent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.adversary = True if i < num_adversaries else False
            agent.speaker = True if i == 2 else False
            agent.movable = False
        
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmakr in enumerate(world.landmarks):
            landmakr.name = 'landmark %d' % i
            landmakr.collide = False
            landmakr.movable = False
        
        # make initial conditions
        self.reset_world(world)

        return world
    
    