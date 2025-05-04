import matplotlib.pyplot as plt
import pandas as pd
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
