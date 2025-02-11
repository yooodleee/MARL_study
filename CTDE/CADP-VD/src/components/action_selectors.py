
import torch
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule


REGISTRY = {}



class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)
    

    def select_action(
            self,
            agent_inputs,
            avail_actions,
            t_env,
            test_mode=False):
        
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions==0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        
        else:
            picked_actions = Categorical(masked_policies).sample().long()
        
        return picked_actions
    

REGISTRY["multinomial"] = MultinomialActionSelector



class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, decay="linear")
        self.epsilon = self.schedule.eval(0)
    

    