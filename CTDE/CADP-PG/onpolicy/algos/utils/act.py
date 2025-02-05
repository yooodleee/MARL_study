from .distribution import Bernoulli, Categorical, DiagGaussian
import torch
import torch.nn as nn



class ACTLayer(nn.Module):
    """
    MLP Module to compute acts.


    Params
    ----------
        action_space: (gym.Space)
            action space.
        inputs_dim: (int)
            dimension of network input.
        use_orthogonal: (bool)
            whether to use orthogonal initialization.
        gain: (float)
            gain of the output layer of the network.
    """

    def __init__(
            self,
            action_space,
            inputs_dim,
            use_orthogoanl,
            gain):
        
        super(ACTLayer, self).__init__()
        self.mixed_action = False
        self.multi_discrete = False

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out = Categorical(
                inputs_dim,
                action_dim,
                use_orthogoanl,
                gain,
            )
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(
                inputs_dim,
                action_dim, use_orthogoanl,
                gain,
            )
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.action_out = Bernoulli(
                inputs_dim,
                action_dim,
                use_orthogoanl,
                gain,
            )
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multi_discrete = True
            action_dims = action_space.high - action_space.low + 1
            self.action_outs = []

            for action_dim in action_dims:
                self.action_outs.append(
                    Categorical(
                        inputs_dim,
                        action_dim,
                        use_orthogoanl,
                        gain,
                    )
                )

            self.action_outs = nn.ModuleList(self.action_outs)
        else:   # discrete + continous
            self.mixed_action = True
            continous_dim = action_space[0].shape[0]
            discrete_dim = action_space[1].n
            self.action_outs = nn.ModuleList(
                [
                    DiagGaussian(
                        inputs_dim,
                        continous_dim,
                        use_orthogoanl,
                        gain,
                    ),
                    Categorical(
                        inputs_dim,
                        discrete_dim,
                        use_orthogoanl,
                        gain,
                    )
                ]
            )
    

    