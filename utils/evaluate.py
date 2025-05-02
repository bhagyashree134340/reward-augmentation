import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from hydra.core.hydra_config import HydraConfig

from utils.plots import plot_validation_stats

log = logging.getLogger(__name__)


def load_rollout_stats(validate_dir):
    """
    Loads all rollout .npz files and extracts returns and lengths.
    Returns two lists: avg_returns, avg_lengths.
    """
    npz_files = sorted(validate_dir.glob("*/rollout_ep*.npz"))
    returns = []
    lengths = []
    timesteps = []

    for file in npz_files:
        data = np.load(file)
        rewards = data["rewards"]
        returns.append(np.sum(rewards))
        lengths.append(len(rewards))
        ep = int(file.stem.split("ep")[-1])
        timesteps.append(ep)

    return timesteps, returns, lengths


def evaluate():
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    validate_dir = output_dir / "validate"

    if not validate_dir.exists():
        raise FileNotFoundError(f"No validation data found in {validate_dir}")

    # Load rollout stats
    timesteps, returns, lengths = load_rollout_stats(validate_dir)

    # Plot stats
    plot_validation_stats(timesteps, returns, lengths, output_dir / "final_evaluation")
