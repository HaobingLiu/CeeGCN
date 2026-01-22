import torch

def alpha_entmax(inputs, alpha = 1.55, dim = 1):
    inputs_max, _ = torch.max(inputs, dim=dim, keepdim=True)
    inputs_exp = torch.exp((inputs - inputs_max) * alpha)
    inputs_sum = torch.sum(inputs_exp, dim=dim, keepdim=True)
    outputs = inputs_exp / inputs_sum
    return outputs
