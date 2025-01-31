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
    

# multi-agent world
class World(object):

    def __init__(self):
        self.agents = []        # list of agents and entities (can change at execution-time!)
        self.landmarks = []     
        self.dim_c = 0          # communication channel dimensionality
        self.dim_p = 2          # position dimensionality
        self.dim_color = 3      # color dimensionality
        self.dt = 0.1           # simulation timestep
        self.daming = 0.25      # physical damping
        self.contact_force = 1e+2   # concat response parameters
        self.contact_margin = 1e-3
    
    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks
    
    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [
            agent for agent in self.agents
            if agent.action_callback is None
        ]
    
    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [
            agent for agent in self.agents
            if agent.action_callback is not None
        ]
    
    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        
        p_force = [None] * len(self.entities)           # gather forces applied to entites
        p_force = self.apply_action_force(p_force)      # apply agent physical controls
        p_force = self.apply_environment_force(p_force) # apply environment forces
        
        self.integrate_state(p_force)   # integrate physical state
        for agent in self.agents:       # update agent state
            self.update_agent_state(agent)
    
    # gather agent action forces
    def apply_action_force(self, p_force):
        for i, agent in enumerate(self.agents): # set applied forces
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
        
        return p_force
    
    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        for a, entity_a in enumerate(self.entities):    # simple (but inefficient) collision response
            for b, entity_b in enumerate(self.entities):
                if (b <= a):
                    continue

                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if (f_a is not None):
                    if (p_force[a] is None):
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                
                if (f_b is not None):
                    if (p_force[b] is None):
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        
        return p_force
    
    # integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue

            entity.state.p_vel = entity.state.p_vel * (1 - self.daming)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel \
                                        / np.sqrt(np.square(
                                            entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])) * entity.max_speed
            
            entity.state.p_pos += entity.state.p_vel * self.dt
    
    def update_agent_state(self, agent):
        if agent.silent:    # set communication state (directly for now)
            agent.state.c = np.zeros(self.dim_c)
        
        else:
            noise = np.random.randn(*agent.action.c.shape) \
                    * agent.c_noise \
                    if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise
    
    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        
        if (entity_a is entity_b):
            return [None, None] # den't collide against itself
        
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = entity_a.size + entity_b.size    # minimum allowable distance

        # softmax penetration
        k = self.contact_margin 
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None

        return [force_a, force_b]