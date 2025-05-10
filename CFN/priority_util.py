import math

import torch
import numpy as np


def compute_cfn_priority(cfn, obs_batch, update_counts, alpha=0.5):
    norms = cfn.compute_output_norm(obs_batch)
    one_over_norms = 1. / (norms + 1e-6)
    one_over_counts = 1. / (update_counts + 1.)

    priorities = alpha * one_over_counts + (1 - alpha) * one_over_norms
    return priorities


def get_coin_flips(coin_flip_dim) -> torch.Tensor:
    """
    function to get the actual coin flip vectors.

    :param coin_flip_dim:
    :return: The random coin flip vectors.
    """

    coin_flips = np.random.choice([-1, 1], size=coin_flip_dim)
    return torch.from_numpy(coin_flips).float()


def compute_intrinsic_reward(coin_flip_d: int, cfn_norm: torch.Tensor) -> float:
    """
    Computes intrinsic reward using coin-flip vector dimensionality and its norm.

    Args:
        coin_flip_d (int): Dimensionality of the coin-flip vector.
        cfn_norm (float): Precomputed squared L2 norm of the vector.

    Returns:
        float: Intrinsic reward value.
"""
    return math.sqrt(cfn_norm.item() / coin_flip_d)



