import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()

        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        world.num_agents = num_agents
        num_adversaries = 1
        num_landmarks = num_agents - 1

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.15
        
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08
        
        # make initial conditions
        self.reset_world(world)

        return world
    
    