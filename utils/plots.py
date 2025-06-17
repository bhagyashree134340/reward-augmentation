from pathlib import Path
import wandb
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

    def to_cpu_scalar_list(x):
        return [v.detach().cpu().item() if torch.is_tensor(v) else v for v in x]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rewards = to_cpu_scalar_list(stats.episode_rewards)
    lengths = to_cpu_scalar_list(stats.episode_lengths)
    timesteps = to_cpu_scalar_list(stats.timesteps_on_ep_end)

    # Episode Return vs. Episode
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.title("Episode Return vs. Episode")
    plt.grid()
    plt.savefig(output_path / f"episode_return_vs_episode_{tag}.png")
    plt.close()

    # Episode Length vs. Episode
    plt.figure(figsize=(10, 5))
    plt.plot(lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length vs. Episode")
    plt.grid()
    plt.savefig(output_path / f"episode_length_vs_episode_{tag}.png")
    plt.close()

    # Episode Return vs. Timesteps
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, rewards)
    plt.xlabel("Environment Timestep")
    plt.ylabel("Episode Return")
    plt.title("Episode Return vs. Environment Timesteps")
    plt.grid()
    plt.savefig(output_path / f"episode_return_vs_timesteps_{tag}.png")
    plt.close()


def plot_eval_curve(env_steps, mean_returns, std_returns, save_path: Path):
    """Plots average return with std band."""
    plt.figure(figsize=(8, 5))

    if torch.is_tensor(env_steps):
        env_steps = env_steps.detach().cpu().numpy()
    else:
        env_steps = np.array(env_steps)

    if torch.is_tensor(mean_returns):
        mean_returns = mean_returns.detach().cpu().numpy()
    else:
        mean_returns = np.array(mean_returns)

    if torch.is_tensor(std_returns):
        std_returns = std_returns.detach().cpu().numpy()
    else:
        std_returns = np.array(std_returns)

    plt.plot(env_steps, mean_returns, label="Mean Return")
    plt.fill_between(env_steps, mean_returns - std_returns, mean_returns + std_returns,
                     alpha=0.3, label="±1 Std. Dev.")
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

    device = next(cfn.parameters()).device

    def compute_novelty(obs, cfn):
        scores = []
        for state in obs:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                score = cfn.compute_squared_output_norm(state_tensor)
                scores.append(score.cpu().item())
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



def log_cfn_stats_to_wandb(cfn, obs, step=None):
    with torch.no_grad():
        obs = obs.to(cfn.device) 
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        combined_out = cfn(obs, update_prior_stats=False)
        prior_out = cfn.prior(obs)
        output_norm = combined_out.norm(p=2, dim=1)
        prior_output_norm = prior_out.norm(p=2, dim=1)
        pseudocount_estimate = cfn.coin_flip_dim/(output_norm**2)

        wandb.log({
            "prior_output_norm": prior_output_norm.cpu().numpy(), 
            "output_norm": output_norm.cpu().numpy(),
            "pseudocount_estimate":pseudocount_estimate.cpu().numpy(),
        }, step=step)
