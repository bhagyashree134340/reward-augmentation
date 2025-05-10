from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

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