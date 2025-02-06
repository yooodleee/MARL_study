
import torch.nn as nn
from .util import init, get_clones


"""
MLP modules.
"""



class MLPLayer(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_size,
            layer_N,
            use_orthogonal,
            use_ReLU):
        
        super(MLPLayer, self).__init__()
        
        self._layer_N = layer_N
        
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(
                m,
                init_method,
                lambda x: nn.init.constant_(x, 0),
                gain=gain
            )
        
        
        self.fc1 = nn.Sequential(
            init_(
                nn.Linear(input_dim, hidden_size)
            ),
            active_func,
            nn.LayerNorm(hidden_size)
        )
        self.fc_h = nn.Sequential(
            init_(
                nn.Linear(hidden_size, hidden_size)
            ),
            active_func,
            nn.LayerNorm(hidden_size)
        )
        self.fc2 = get_clones(self.fc_h, self._layer_N)
    

    def forward(self, x):
        x = self.fc1(x)
        
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        
        return x
    


