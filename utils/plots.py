from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

import torch

log = logging.getLogger(__name__)


def plot_training_stats(plot_path, stats, smoothing_window=20):
    """
    Plots episode length and smoothed reward over time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

    axes[0].plot(stats.episode_lengths)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode Length")
    axes[0].set_title("Episode Length over Time")

    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(
        smoothing_window, min_periods=smoothing_window
    ).mean()
    axes[1].plot(rewards_smoothed)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Reward (Smoothed)")
    axes[1].set_title(
        f"Episode Reward over Time\n(Smoothed over window size {smoothing_window})"
    )

    fig.savefig(plot_path)
    plt.close(fig)
    log.info(f"Saved training plots to {plot_path}")


def plot_rollout_rewards(plot_path, rewards, smoothing_window=20):
    """
    Plots validation return and episode length across validation episodes.
    """
    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

    rewards_smoothed = pd.Series(rewards).rolling(
        smoothing_window, min_periods=smoothing_window
    ).mean()
    ax.plot(rewards_smoothed)
    ax.set_xlabel("Episode time steps")
    ax.set_ylabel("Episode Reward (Smoothed)")
    ax.set_title(
        f"Episode Reward over Time\n(Smoothed over window size {smoothing_window})"
    )

    fig.savefig(plot_path)
    plt.close(fig)
    log.info(f"Saved training plots to {plot_path}")


def plot_validation_stats(timesteps, returns, lengths, output_dir):
    """
    Plots return and episode length vs training timestep.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

    axes[0].plot(timesteps, returns)
    axes[0].set_xlabel("Training Episode")
    axes[0].set_ylabel("Return")
    axes[0].set_title("Average Return vs Training Episode")

    axes[1].plot(timesteps, lengths)
    axes[1].set_xlabel("Training Episode")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_title("Episode Length vs Training Episode")

    plot_path = output_dir / "evaluation_summary.png"
    fig.savefig(plot_path)
    plt.close(fig)
    log.info(f"Saved evaluation plots to {plot_path}")


def plot_and_save_training_metrics(stats, output_dir, tag="default"):
    """
    Plots and saves training metrics as PNGs.

    :param stats: EpisodeStats namedtuple with episode_lengths, episode_rewards, timesteps_on_ep_end
    :param output_dir: Path to output directory.
    :param tag: Optional tag to distinguish saved plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Episode Return vs. Episode
    plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title("Episode Return vs. Episode")
    plt.grid()
    plt.savefig(output_path / f"episode_return_vs_episode_{tag}.png")
    plt.close()

    # Episode Length vs. Episode
    plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length vs. Episode")
    plt.grid()
    plt.savefig(output_path / f"episode_length_vs_episode_{tag}.png")
    plt.close()

    # Episode Return vs. Timesteps
    plt.figure(figsize=(10, 5))
    plt.plot(stats.timesteps_on_ep_end, stats.episode_rewards)
    plt.xlabel("Environment Timestep")
    plt.ylabel("Episode Return")
    plt.title("Episode Return vs. Environment Timesteps")
    plt.grid()
    plt.savefig(output_path / f"episode_return_vs_timesteps_{tag}.png")
    plt.close()


def plot_eval_curve(env_steps, mean_returns, std_returns, save_path: Path):
    """Plots average return with std band."""
    plt.figure(figsize=(8, 5))
    env_steps = np.array(env_steps)
    mean_returns = np.array(mean_returns)
    std_returns = np.array(std_returns)

    plt.plot(env_steps, mean_returns, label="Mean Return")
    plt.fill_between(env_steps, mean_returns - std_returns, mean_returns + std_returns, alpha=0.3, label="±1 Std. Dev.")
    plt.xlabel("Environment Steps")
    plt.ylabel("Average Return")
    plt.title("Evaluation Return During Training")
    plt.legend()
    plt.grid(True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_return_distributions(return_dict, save_path):
    """
    Plot the return distributions from multiple evaluation points.

    :param return_dict: dict of {env_step: list_of_returns}
    :param save_path: where to save the final plot
    """
    plt.figure(figsize=(10, 6))

    for step, returns in sorted(return_dict.items()):
        plt.hist(returns, bins=20, alpha=0.6, label=f"Step {step}", density=True)

    plt.xlabel("Episode Return")
    plt.ylabel("Density")
    plt.title("Return Distributions Across Training Steps")
    plt.legend()
    plt.grid(True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


# def cfn_early_vs_late_training_comparison(cfn, eval_dir):
#     # Get all evaluation files sorted by step
#     eval_files = sorted(Path(eval_dir).glob("evaluation_step*.npz"),
#                         key=lambda x: int(x.stem.split("step")[-1]))
#
#     # Load first 7 and last 7 files
#     first_files = eval_files[:7]
#     last_files = eval_files[-7:]
#
#     # Load observations from all files
#     def load_obs_from_files(files):
#         all_obs = []
#         for file in files:
#             data = np.load(file)
#             all_obs.append(data["observations"])
#         return np.concatenate(all_obs)
#
#     first_obs = load_obs_from_files(first_files)
#     last_obs = load_obs_from_files(last_files)
#
#     # Compute novelty scores
#     def compute_novelty(obs, cfn):
#         scores = []
#         for state in obs:
#             state_tensor = torch.FloatTensor(state).unsqueeze(0)
#             with torch.no_grad():
#                 score = cfn.compute_squared_output_norm(state_tensor).item()
#             scores.append(score)
#         return np.array(scores)
#
#     first_scores = compute_novelty(first_obs, cfn)
#     last_scores = compute_novelty(last_obs, cfn)
#
#     # Calculate means and stds
#     first_mean, first_std = np.mean(first_scores), np.std(first_scores)
#     last_mean, last_std = np.mean(last_scores), np.std(last_scores)
#
#     # Plot comparison
#     plt.figure(figsize=(8, 6))
#     bars = plt.bar(
#         ["First 7 Evals (Early)", "Last 7 Evals (Late)"],
#         [first_mean, last_mean],
#         yerr=[first_std, last_std],
#         capsize=10,
#         color=["skyblue", "salmon"],
#         alpha=0.7
#     )
#
#     # Add value labels
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2., height,
#                  f"{height:.2f} ± {first_std if 'First' in bar.get_label() else last_std:.2f}",
#                  ha='center', va='bottom')
#
#     plt.ylabel("Mean Novelty Score (‖fϕ(s)‖²)")
#     plt.title("CFN Novelty Comparison: Early vs. Late Training (7 Eval Runs Each)")
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.savefig(Path(eval_dir).parent / "cfn_novelty_comparison.png")
#
#     log.info(f"Early Training - Mean: {first_mean:.2f} ± {first_std:.2f}")
#     log.info(f"Late Training - Mean: {last_mean:.2f} ± {last_std:.2f}")


def cfn_early_vs_late_training_comparison(cfn, eval_dir):
    eval_files = sorted(Path(eval_dir).glob("evaluation_step*.npz"),
                        key=lambda x: int(x.stem.split("step")[-1]))

    first_files = eval_files[:7]
    last_files = eval_files[-7:]

    def load_obs_from_files(files):
        all_obs = []
        for file in files:
            data = np.load(file)
            all_obs.append(data["observations"])
        return np.concatenate(all_obs)

    first_obs = load_obs_from_files(first_files)
    last_obs = load_obs_from_files(last_files)

    def compute_novelty(obs, cfn):
        scores = []
        for state in obs:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                score = cfn.compute_squared_output_norm(state_tensor).item()
            scores.append(score)
        return np.array(scores)

    first_scores = compute_novelty(first_obs, cfn)
    last_scores = compute_novelty(last_obs, cfn)

    first_mean, first_std = np.mean(first_scores), np.std(first_scores)
    last_mean, last_std = np.mean(last_scores), np.std(last_scores)

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        ["First 7 Evals (Early)", "Last 7 Evals (Late)"],
        [first_mean, last_mean],
        yerr=[first_std, last_std],
        capsize=10,
        color=["skyblue", "salmon"],
        alpha=0.7
    )

    for i, bar in enumerate(bars):
        height = bar.get_height()
        label_std = first_std if i == 0 else last_std
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f"{height:.2f} ± {label_std:.2f}",
                 ha='center', va='bottom')

    plt.ylabel("Mean (‖fϕ(s)‖²)")
    plt.title("CFN Novelty Comparison: Early vs. Late Training (7 Eval Runs Each)")
    plt.grid(True, linestyle='--', alpha=0.3)

    save_path = Path(eval_dir).parent / "cfn_novelty_comparison.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    log.info(f"Plot saved to: {save_path}")
    log.info(f"Early Training - Mean: {first_mean:.2f} ± {first_std:.2f}")
    log.info(f"Late Training - Mean: {last_mean:.2f} ± {last_std:.2f}")
