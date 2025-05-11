import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

import torch
import wandb
from hydra.core.hydra_config import HydraConfig

from utils.gif import evaluate_policy
from utils.plots import plot_validation_stats

log = logging.getLogger(__name__)


def evaluate_and_log(actor, env, current_timestep, max_steps):
    # Evaluate policy
    eval_data = evaluate_policy(actor, env, num_episodes=10, max_steps=max_steps)

    mean_r = np.mean(eval_data["episode_rewards"])
    std_r = np.std(eval_data["episode_rewards"])
    # self.eval_returns_by_step[current_timestep] = eval_data["episode_rewards"]

    # Save actor checkpoint
    actor_path = Path(
        HydraConfig.get().runtime.output_dir) / "checkpoints" / f"sac_actor_step{current_timestep}.pt"
    actor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), actor_path)

    # Save evaluation data (.npz)
    eval_dir = Path(HydraConfig.get().runtime.output_dir) / "evaluate"
    eval_dir.mkdir(parents=True, exist_ok=True)
    np.savez(eval_dir / f"evaluation_step{current_timestep}.npz", **eval_data)

    # Log to WandB
    wandb.log({
        "eval_mean_return": mean_r,
        # "eval_std_return": std_r,
    }, step=current_timestep)

    # Save data for plotting
    return current_timestep, mean_r, std_r


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
