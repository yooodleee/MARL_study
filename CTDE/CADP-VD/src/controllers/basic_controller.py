
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY


import torch



class BasicMAC:
    """
    This multi-agent controller shares params between agents.
    """

    def __init__(
            self,
            scheme,
            groups,
            args):
        
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
    

    def select_actions(
            self,
            ep_batch,
            t_ep,
            t_env,
            bs=slice(None),
            test_mode=False):
        
        """
        Only select acts for the selected batch elements in bs.
        """

        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs],
                                                            avail_actions[bs],
                                                            t_env,
                                                            test_mode=test_mode)
        
        return chosen_actions
    

    