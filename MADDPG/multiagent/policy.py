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
    
    def action(self, obs):
        """
        Ignore observation and just act based on keyboard events.
        """
        if self.env.discrete_action_input:
            u = 0
            if self.move[0]: u = 1
            if self.move[1]: u = 2
            if self.move[2]: u = 4
            if self.move[3]: u = 3
        
        else:
            u = np.zeros(5) # 5-d because of no-move action
            if self.move[0]: u[1] += 1.0
            if self.move[1]: u[2] += 1.0
            if self.move[3]: u[3] += 1.0
            if self.move[2]: u[3] += 1.0
            if True not in self.move:
                u[0] += 1.0
        
        return np.concatenate([u, np.zeros(self.env.world.dim_c)])
    
    