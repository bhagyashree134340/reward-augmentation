import matplotlib.pyplot as plt
import pandas as pd
import logging

log = logging.getLogger(__name__)


def plot_training_stats(plot_path, stats, smoothing_window=20):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

    axes[0].plot(stats.episode_lengths)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode Length")
    axes[0].set_title("Episode Length over Time")

    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    axes[1].plot(rewards_smoothed)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Reward (Smoothed)")
    axes[1].set_title(f"Episode Reward over Time\n(Smoothed over window size {smoothing_window})")

    fig.savefig(plot_path)
    plt.close(fig)
    log.info(f"Saved training plots to {plot_path}")
