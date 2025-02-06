
import torch.nn as nn
from .util import init


"""
CNN Modules and utils.
"""


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)



class CNNLayer(nn.Module):

    def __init__(
            self,
            obs_shape,
            hidden_size,
            use_orthogonal,
            use_ReLU,
            kernel_size=3,
            stride=1):
        
        super(CNNLayer, self).__init__()

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
        
        input_channel = obs_shape[0]
        input_width = obs_shape[1]
        input_height = obs_shape[2]

        self.cnn = nn.Sequential(
            init_(
                nn.Conv2d(
                    in_channels=input_channel,
                    out_channels=hidden_size // 2,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            ),
            active_func,
            Flatten(),
            init_(
                nn.Linear(
                    hidden_size // 2 \
                    * (input_width - kernel_size + stride) \
                    * (input_height - kernel_size + stride),
                    hidden_size
                )
            ),
            active_func,
            init_(
                nn.Linear(
                    hidden_size,
                    hidden_size,
                )
            ),
            active_func,
        )

    
    