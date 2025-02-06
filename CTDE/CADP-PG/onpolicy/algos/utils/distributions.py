
import torch
import torch.nn as nn
from .util import init


"""
Modify standard PyTorch distributions so they to make compatible
    with this codebase.
"""


# 
# Standardize distribution interfaces
#



# Categorical
class FixedCategorical(torch.distributions.Categorical):

    def sample(self):
        return super().sample().unsqueeze(-1)
    

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )
    
    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)



# Normal
class FixedNormal(torch.distributions.Normal):

    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)
    

    def entropy(self):
        return self.entropy().sum(-1)
    

    def mode(self):
        return self.mean



# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):

    def log_probs(self, actions):
        return super\
            .log_prob(actions)\
            .view(actions.size(0), -1)\
            .sum(-1)\
            .unsqueeze(-1)
    

    def entropy(self):
        return super().entropy().sum(-1)
    

    def mode(self):
        return torch.gt(self.probs, 0.5).float()



class Categorical(nn.Module):

    def __init__(
            self,
            num_inputs,
            num_outputs,
            use_orthogonal=True,
            gain=0.01):
        
        super(Categorical, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        
        def init_(m):
            return init(
                m,
                init_method,
                lambda x: nn.init.constant_(x, 0),
                gain
            )
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))


    def forward(
            self,
            x,
            available_actions=None):
        
        x = self.linear(x)

        if available_actions is not None:
            x[available_actions == 0] = -1e10
        
        return FixedCategorical(logits=x)



