
import numpy as np
import math
import torch


def get_params_size(params_list):
    # params_size = sum([np.prod(p.size()) for p in params_list]) * 4 / 1024
    # return "{:.0f}KB".format(params_size)

    params_size = sum([np.prod(list(p.size())) for p in params_list]) / 1000

    return "{:.0f}K".format(params_size)


