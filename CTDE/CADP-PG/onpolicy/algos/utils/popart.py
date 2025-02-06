
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class PopArt(torch.nn.Module):

    def __init__(
            self,
            input_shape,
            output_shape,
            norm_axes=1,
            beta=0.99999,
            epsilon=1e-5,
            device=torch.device("cpu")):
        
        super(PopArt, self).__init__()

        self.beta = beta
        self.epsilon = epsilon
        self.norm_axes = norm_axes
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape)).to(**self.tpdv)
        self.bias = nn.Parameter(torch.Tensor(output_shape)).to(**self.tpdv)

        self.stddev = nn.Parameter(torch.ones(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.mean_sq = nn.Parameter(torch.zeros(output_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

        self.reset_parameters()


    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weight
            )
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        
        self.mean.zero_()
        self.mean_sq.zero_()
        self.debiasing_term.zero_()
    

    def forward(self, input_vector):

        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        
        input_vector = input_vector.to(**self.tpdv)

        return F.linear(
            input_vector,
            self.weight,
            self.bias
        )
    

    