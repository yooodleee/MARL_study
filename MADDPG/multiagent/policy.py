import numpy as np
from pyglet.window import key


class Policy(object):
    """
    Individual agent policy.
    """

    def __init__(self):
        pass

    def action(self, obs):
        raise NotImplementedError()


class InteractivePolicy(Policy):
    """
    Interactive policy based on keyboard input.
    hard-coded to deal only with movement, not communication.
    
    hard-coded keyboard events
    -----------------------------
    move
    comm

    register keyboard events with this environment's window
    ---------------------------------------------------------
    env.viewers[agent_index].window.on_key_press
    env.viewers[agent_index].window.on_key_release
    """

    def __init__(self, env, agent_index):
        super(InteractivePolicy, self).__init__()

        self.env = env
        self.move = [False for i in range(4)]
        self.comm = [False for i in range(env.world.dim_c)]

        env.viewers[agent_index].window.on_key_press = self.key_press
        env.viewers[agent_index].window.on_key_release = self.key_release
    
    