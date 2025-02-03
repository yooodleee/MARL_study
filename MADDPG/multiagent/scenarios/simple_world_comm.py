
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()

        # set any world properties first
        world.dim_c = 4
        # world.damping = 1
        num_good_agents = 2
        num_adversaries = 4
        num_agents = num_adversaries + num_good_agents
        num_landmarks = 1
        num_food = 2
        num_forests = 2
        
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.leader = True if i == 0 else False
            agent.silent = True if i > 0 else False
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.045
            agent.accel = 3.0 if agent.adversary else 4.0
            # agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
        
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        
        world.food = [Landmark() for i in range(num_food)]
        for i, landmark in enumerate(world.food):
            landmark.name = 'food %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.03
            landmark.bondary = False
        
        world.forests = [Landmark() for i in range(num_forests)]
        for i, landmar in enumerate(world.forests):
            landmark.name = 'forest %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.3
            landmark.boundary = False
        
        world.landmarks += world.food
        world.landmarks += world.forests
        # world.landmarks += self.set_boundaries(world) 
        # world boundaries now penalize with negative reward

        # make initial conditions
        self.reset_world(world)
        return world
    

    def set_boundaries(self, world):
        boundary_list = []
        landmark_size = 1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)

        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(l)
        
        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(l)
        
        for i, l in enumerate(boundary_list):
            l.name = 'boundary %d' % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)
        
        return boundary_list
    

    def reset_world(self, world):
        
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array(
                [0.45, 0.95, 0.45]
            ) if not agent.adversary else np.array(
                [0.95, 0.45, 0.45]
            )
            agent.color -= np.array(
                [0.3, 0.3, 0.3]
            ) if agent.leader else np.array([0, 0, 0])

            # random properties for landmarks
            for i, landmark in enumerate(world.landmarks):
                landmark.color = np.array([0.25, 0.25, 0.25])
            
            for i, landmark in enumerate(world.food):
                landmark.color = np.array([0.15, 0.15, 0.65])
            
            for i, landmark in enumerate(world.forests):
                landmark.color = np.array([0.6, 0.9, 0.6])
            
            # set random initial states
            for agent in world.agents:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
            
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
            
            for i, landmark in enumerate(world.food):
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
            
            for i, landmark in enumerate(world.forests):
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.stat.p_vel = np.zeros(world.dim_p)
        

    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
                    
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size

        return True if dist < dist_min else False
        

    