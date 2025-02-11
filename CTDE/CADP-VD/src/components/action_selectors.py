
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
    

    