import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()

        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_adversaries = 1
        num_landmarks = 2

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True

            if i < num_adversaries:
                agent.adversary = True
            else:
                agent.adversary = False
        
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        
        # make initial conditions
        self.reset_world(world)
        return world
    
    