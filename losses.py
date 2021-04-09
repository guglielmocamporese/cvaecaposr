##################################################
# Imports
##################################################

import torch
import numpy as np

# KL Divergence
def kl_div(p_mean, p_var, t_mean, t_var):
    """
    Compute the KL-Divergence between two Gaussians p and q:
        p ~ N(p_mean, diag(p_var))
        t ~ N(t_mean, diag(t_var))

    Args:
        p_mean: tensor of shape [bs(, ...), dim]
        p_var: tensor of shape [bs(, ...), dim]
        t_mean: tensor of shape [bs(, ...), dim]
        t_var: tensor of shape [bs(, ...), dim]

    Output:
        kl: tensor of shape [bs(, ...)]
    """
    if torch.is_tensor(p_mean):
        kl = - 0.5 * (torch.log(p_var) - torch.log(t_var) + 1 - p_var / t_var - (p_mean - t_mean).pow(2) / t_var ).sum(-1)
    else:
        kl = - 0.5 * (np.log(p_var) - np.log(t_var) + 1 - p_var / t_var - ((p_mean - t_mean) ** 2) / t_var).sum(-1)
    return kl
