import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

import torch
import wandb
from hydra.core.hydra_config import HydraConfig

from CFN.priority_util import compute_intrinsic_reward
from utils.gif import evaluate_policy
from utils.plots import plot_validation_stats

log = logging.getLogger(__name__)


# TODO: add an output dir
from pathlib import Path
import numpy as np
import wandb
from hydra.core.hydra_config import HydraConfig


def evaluate(actor, env, current_timestep, max_steps, path=None):
    env.training = True

    eval_data = evaluate_policy(actor, env, num_episodes=10, max_steps=max_steps)

    mean_r = np.mean(eval_data["episode_rewards"])
    std_r = np.std(eval_data["episode_rewards"])

    if path is None:
        eval_dir = Path(HydraConfig.get().runtime.output_dir) / "evaluate"
    else:
        eval_dir = Path(path)
    eval_dir.mkdir(parents=True, exist_ok=True)

    np.savez(eval_dir / f"evaluation_step{current_timestep}.npz", **eval_data)

    wandb.log({
        "eval_mean_return": mean_r,
        # "eval_std_return": std_r,
    }, step=current_timestep)

    env.training = False

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


# def evaluate():
#     output_dir = Path(HydraConfig.get().runtime.output_dir)
#     validate_dir = output_dir / "validate"
#
#     if not validate_dir.exists():
#         raise FileNotFoundError(f"No validation data found in {validate_dir}")
#
#     # Load rollout stats
#     timesteps, returns, lengths = load_rollout_stats(validate_dir)
#
#     # Plot stats
#     plot_validation_stats(timesteps, returns, lengths, output_dir / "final_evaluation")


def evaluate_cfn_bonus_generalization(cfn, env, buffer, num_samples=100):
    """

    :param num_samples:
    :param cfn:
    :param env:
    :param seen_obs:
    :param device:
    :param num_unseen:
    :return:
    """

    # TODO: should i set it to eval mode?

    # seen obs from the buffer?
    seen_obs = buffer.sample(batch_size=num_samples)["obs"]

    # Sample unseen observations from the env
    unseen_obs = [env.observation_space.sample() for _ in range(num_samples)]
    seen_tensor = torch.tensor(seen_obs, dtype=torch.float32)
    unseen_tensor = torch.tensor(unseen_obs, dtype=torch.float32)

    seen_score = compute_intrinsic_reward(16, cfn.compute_squared_output_norm(seen_tensor))
    unseen_score = compute_intrinsic_reward(16, cfn.compute_squared_output_norm(unseen_tensor))

    seen_norm_mean = seen_score.mean().item()
    seen_norm_std = seen_score.std().item()
    unseen_norm_mean = unseen_score.mean().item()
    unseen_norm_std = unseen_score.std().item()

    labels = ['Seen', 'Unseen']
    means = [seen_norm_mean, unseen_norm_mean]
    stds = [seen_norm_std, unseen_norm_std]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        labels,
        means,
        yerr=stds,
        capsize=10,
        color=['skyblue', 'salmon'],
        alpha=0.7
    )

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2., height,
            f"{height:.2f} ± {stds[i]:.2f}",
            ha='center', va='bottom'
        )

    plt.ylabel("Mean (‖fϕ(s)‖²)")
    plt.title("CFN Bonus Generalization: Seen vs. Unseen Observations")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "cfn_bonus_generalization_bar.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    import logging
    log = logging.getLogger(__name__)
    log.info(f"Plot saved to: {plot_path}")
    log.info(f"Seen - Mean: {seen_norm_mean:.2f} ± {seen_norm_std:.2f}")
    log.info(f"Unseen - Mean: {unseen_norm_mean:.2f} ± {unseen_norm_std:.2f}")
