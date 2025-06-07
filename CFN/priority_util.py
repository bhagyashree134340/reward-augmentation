import math

import torch
import numpy as np
from torch import Tensor


def compute_cfn_priority(cfn, obs_batch, update_counts, coin_flip_dim, alpha=0.5):
    norms = cfn.compute_squared_output_norm(obs_batch)
    scaled_norms = norms / coin_flip_dim
    # TODO: cant i add like 0.001 instead of 1.0 to avoid dividing by 0?
    one_over_counts = 1.0 / (update_counts + 1.0)
    priorities = alpha * one_over_counts + (1 - alpha) * scaled_norms

    return priorities


def get_coin_flips(coin_flip_dim) -> torch.Tensor:
    """
    function to get the actual coin flip vectors.

    :param coin_flip_dim:
    :return: The random coin flip vectors.
    """

    coin_flips = np.random.choice([-1, 1], size=coin_flip_dim)
    return torch.from_numpy(coin_flips).float()


def compute_intrinsic_reward(coin_flip_d: int, cfn_norm: torch.Tensor) -> Tensor:
    """
    Computes intrinsic reward using coin-flip vector dimensionality and its norm.

    Args:
        coin_flip_d (int): Dimensionality of the coin-flip vector.
        cfn_norm (float): Precomputed squared L2 norm of the vector.

    Returns:
        float: Intrinsic reward value.
"""
    # torch.sqrt(cfn_squared_norms / coin_flip_d)? i dont think it's necessary
    return torch.sqrt(cfn_norm / coin_flip_d)
