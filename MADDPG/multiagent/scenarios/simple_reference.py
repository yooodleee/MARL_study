
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()

        # set any world properties first
        world.dim_c = 10
        world.collaborative = True  # whether agents share rewards

        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
        
        # add landmarks
        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        
        # make initial conditions
        self.reset_world(world)
        return world
    

    def reset_world(self, world):
        
        # assign goals to agents
        for agent in world.agents:
            agent.goal_a = None
            agent.goal_b = None
        
        # want other agent to go to the goal landmark
        world.agents[0].goal_a = world.agents[1]
        world.agents[0].goal_b = np.random.choice(world.landmarks)
        world.agents[1].goal_a = world.agents[0]
        world.agents[1].goal_b = np.random.choice(world.landmarks)

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.75, 0.25, 0.25])
        world.landmarks[1].color = np.array([0.25, 0.75, 0.25])
        world.landmarks[2].color = np.array([0.25, 0.25, 0.75])

        # special color for goals
        world.agents[0].goal_a.color = world.agents[0].goal_b.color
        world.agents[1].goal_a.color = world.agents[1].goal_b.color

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
    

    