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


# properties and state of physical world entity
class Entity(object):

    def __init__(self):
        self.name = ''              # name
        self.size = 0.050           # properties
        self.movable = False        # entity can move / be pushed
        self.collide = True         # entity collides with others
        self.density = 25.0         # material density (affects mass)
        self.color = None           # color
        self.max_speed = None       # max speed
        self.accel = None           # accel
        self.state = EntityState()  # state
        self.initial_mass = 1.0     # mass
    
    @property
    def mass(self):
        return self.initial_mass


# properties of landmark entities
class Landmark(Entity):

    def __init__(self):
        super(Landmark, self).__init__()


# properties of agent entities
class Agent(Entity):

    def __init__(self):
        super(Agent, self).__init__()

        self.movable = True         # agents are movable by default
        self.silent = False         # cannot send communication signals
        self.blind = False          # cannot observe the world
        self.u_noise = None         # physical motor noise amount
        self.c_noise = None         # communication noise amount
        self.u_range = 1.0          # control range
        self.state = AgentState()   # state
        self.action = Action()      # action
        self.action_callback = None # script behavior to execute
    

