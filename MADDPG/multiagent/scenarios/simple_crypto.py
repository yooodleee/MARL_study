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
    
    
    def reset_world(self, world):
        
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
            
            agent.key = None
        
        # random properties for landmarks
        color_list = [np.zeros(world.dim_c) for i in world.landmarks]
        for i, color in enumerate(color_list):
            color[i] += 1
        
        for color, landmark in zip(color_list, world.landmarks):
            landmark.color = color
        
        # set gaol landmark
        goal = np.random.choice(world.landmarks)
        world.agents[1].color = goal.color
        world.agents[2].key = np.random.choice(world.landmarks).color

        for agent in world.agents:
            agent.goal_a = goal
        
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.wim_p)
            agent.state.c = np.zeros(world.dim_c)
        
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
    

    def benchmark_data(self, agent, world):
        """
        Returns data for benchmarking purposes
        """
        return (agent.state.c, agent.goal_a.color)
    
    def good_listeners(self, world):
        """
        Return all agents that are not adversaries
        """
        return [
            agent for agent in world.agents
            if not agent.adversary and not agent.speaker
        ]
    
    