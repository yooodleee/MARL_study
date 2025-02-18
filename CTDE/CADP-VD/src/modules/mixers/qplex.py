
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
    

    def calc_adv(
            self,
            agent_qs,
            states,
            actions,
            max_q_i,
    ):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).clone().detach()

        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)


        if self.args.is_minus_one:
            adv_tot = torch.sum(adv_q * (adv_w_final - 1.), dim=1)
        else:
            adv_tot = torch.sum(adv_q * adv_w_final, dim=1)
        
        return adv_tot
    

    def calc(
            self,
            agent_qs,
            states,
            actions=None,
            max_q_i=None,
            is_v=False,
    ):
        if is_v:
            v_tot = self.calc_v(agent_qs)
            return v_tot
        else:
            adv_tot = self.calc_adv(agent_qs, states, actions, max_q_i)

            return adv_tot
    

    def forward(
            self,
            agent_qs,
            states,
            actions=None,
            max_q_i=None,
            is_v=False,
    ):
        bs = agent_qs.size(0)
        # agent_qs.retain_grad()
        # global_Grad.x = agent_qs

        w_final, v, attend_mag_regs, head_entropies = self.attention_weight(
            agent_qs, states, actions
        )
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v /= self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            max_q_i = w_final * max_q_i + v
        
        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot, attend_mag_regs, head_entropies



class Qatten_Weight(nn.Module):

    def __init__(self, args):
        super(Qatten_Weight, self).__init__()

        self.name = "qatten_weight"
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.unit_dim = args.unit_dim
        self.n_actions = args.n_actions
        self.sa_dim = self.state_dim + self.n_agents * self.n_actions
        self.n_head = args.n_head   # attention head num

        self.embed_dim = args.mixing_embed_dim
        self.attend_reg_coef = args.attend_reg_coef

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()

        hypernet_embed = self.args.hypernet_embed
        for i in range(self.n_head):    # multi-head attention
            selector_nn = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim, bias=False),
            )
            self.selector_extractors.append(selector_nn)    # query

            if self.args.nonlinear: # add qs
                self.key_extractors.append(nn.Linear(self.unit_dim + 1, self.embed_dim, bias=False))   # key
            else:
                self.key_extractors.append(nn.Linear(self.unit_dim, self.embed_dim, bias=False))   # key
        
        if self.args.weighted_head:
            self.hyper_w_head = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.n_head),
            )
        

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )
    

    