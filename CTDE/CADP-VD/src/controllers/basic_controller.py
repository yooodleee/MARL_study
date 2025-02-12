
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
    

    def forward(
            self,
            ep_batch,
            t,
            test_mode=False):
        
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)


        # softmax the agent outputs if they're policy logits.
        if self.agent_output_type == "pi_logits":
            if getattr(
                self.args,
                "mask_before_softmax",
                True
            ):
                # make the logits for unavailable acts very negative to minimise 
                # their affect on the softmax.
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            
            
            agent_outs = torch.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)

                if getattr(
                    self.args,
                    "mask_before_softmax",
                    True
                ):
                    # with prob eps, pick an available act uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=-1, keepdim=True).float()
                
                
                agent_outs = (
                    (1 - self.action_selector.epsilon) 
                    * agent_outs
                    + torch.ones_like(agent_outs)
                    * self.action_selector.epsilon 
                    / epsilon_action_num
                )

                if getattr(
                    self.args,
                    "mask_before_softmax",
                    True
                ):
                    # zero out the unavailable acts
                    agent_outs[reshaped_avail_actions == 0] = 0.0
        

        return agent_outs.view(
            ep_batch.batch_size, self.n_agents, -1
        )
    

    