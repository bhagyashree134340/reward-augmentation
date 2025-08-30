import math

import torch
import numpy as np
from torch import Tensor


def compute_cfn_priority(cfn, obs_batch: Tensor, update_counts: Tensor, coin_flip_dim: int, alpha=0.5, device=None) -> Tensor:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_batch = obs_batch.to(device)
    update_counts = update_counts.to(device)

    norms = cfn.compute_squared_output_norm(obs_batch)
    scaled_norms = norms / coin_flip_dim

    # Add a small epsilon to avoid division by zero (better than 1.0)
    epsilon = 1e-3
    one_over_counts = 1.0 / (update_counts + epsilon)

    one_over_counts = one_over_counts.to(device)

    priorities = alpha * one_over_counts + (1 - alpha) * scaled_norms
    return priorities


def get_coin_flips(coin_flip_dim: int, device=None) -> torch.Tensor:
    """
    Generate a random coin flip vector with values -1 or 1.

    Args:
        coin_flip_dim (int): Dimensionality of the coin flip vector.
        device (torch.device): Device to place the tensor on.

    Returns:
        torch.Tensor: Random coin flip vector of shape (coin_flip_dim,)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coin_flips = np.random.choice([-1, 1], size=coin_flip_dim)
    return torch.from_numpy(coin_flips).float().to(device)


def compute_intrinsic_reward(coin_flip_d: int, cfn_norm: Tensor, device=None) -> Tensor:
    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.sqrt(cfn_norm / coin_flip_d).to(device)