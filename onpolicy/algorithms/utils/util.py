import copy

import numpy as np
import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def print_modules_and_grad_status(model):
    for name, module in model.named_modules():
        print(f"Module: {name}")
        for param_name, param in module.named_parameters(recurse=False):
            print(f"  {param_name} requires_grad: {param.requires_grad}")