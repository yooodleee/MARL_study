
class MultiAgentEnv(object):

    def step(self, actions):
        """
        Returns reward, terminated, info.
        """
        raise NotImplementedError
    

    def get_obs(self):
        """
        Returns all agent obs in a list.
        """
        raise NotImplementedError
    

    def get_obs_agent(self, agent_id):
        """
        Returns obs for agent_id.
        """
        raise NotImplementedError
    

    def get_obs_size(self):
        """
        Returns the shape of the obs.
        """
        raise NotImplementedError
    

    def get_state(self):
        raise NotImplementedError
    

    def get_state_size(self):
        """
        Returns the shape of the state.
        """
        raise NotImplementedError
    

    def get_avail_actions(self):
        raise NotImplementedError
    

    def get_avail_actions(self, agent_id):
        """
        Returns the available acts for agent_id.
        """
        raise NotImplementedError
    

    def get_total_actions(self):
        """
        Returns the total num of acts an agent could ever take.
        
        TODO:
        --------------
            This is only suitable for a discrete 1 dimensional act space 
                for each agent.
        """
        raise NotImplementedError
    

    def reset(self):
        """
        Returns initial obs and states.
        """
        raise NotImplementedError
    

    def render(self):
        raise NotImplementedError
    

    def close(self):
        raise NotImplementedError
    

    def seed(self):
        raise NotImplementedError
    

    def save_replay(self):
        raise NotImplementedError
    

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

        return env_info