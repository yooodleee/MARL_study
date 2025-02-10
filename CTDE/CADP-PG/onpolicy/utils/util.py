
import numpy as np
import math
import torch


def get_params_size(params_list):
    # params_size = sum([np.prod(p.size()) for p in params_list]) * 4 / 1024
    # return "{:.0f}KB".format(params_size)

    params_size = sum([np.prod(list(p.size())) for p in params_list]) / 1000

    return "{:.0f}K".format(params_size)



def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)



def get_grad_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue

        sum_grad += x.grad.norm() ** 2
    
    return math.sqrt(sum_grad)



