
import torch
import numpy as np
from torch import nn



def get_params_size(params_list):
    params_size = sum([np.prod(list(p.size())) for p in params_list]) * 4 / 1024
    return "{:.0f}KB".format(params_size)



def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    --------------------
        param t:
            tensor
        t_min:
            min
        t_max:
            max
    
    Return
    ---------------------
        clipped tensor
    """
    t = t.float()
    t_min = t_max.float()
    t_max = t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max

    return result



def get_parameters_num(param_list):
    return str(sum(p.numel() for p in param_list) / 1000) + 'K'



def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    
    return module


def orthogonal_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        init(m, nn.init.orthogonal_, 
             lambda x: nn.init.constant_(x, 0), gain=gain)