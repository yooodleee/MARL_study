
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class DMAQ_QattenMixer(nn.Module):

    def __init__(self, args):
        super(DMAQ_QattenMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1

        self.attention_weight = Qatten_Weight(args)
        self.si_weight = DMAQ_SI_Weight(args)
    

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        v_tot = torch.sum(agent_qs, dim=-1)

        return v_tot
    

    