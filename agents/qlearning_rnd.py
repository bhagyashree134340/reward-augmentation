from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import imageio
import gymnasium as gym
from cpprb import ReplayBuffer
from gymnasium.wrappers.utils import RunningMeanStd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import wandb
import random

from RND.rnd import RNDModel, RewardForwardFilter


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def one_hot(state, size):
    vec = torch.zeros(size)
    vec[state] = 1.0

    return vec.unsqueeze(0)


def train_q_learning(env, rnd,
                     rnd_optimizer,
                     obs_rms,
                     reward_rms,
                     reward_filter,
                     max_timesteps,
                     alpha, gamma, epsilon, epsilon_decay, epsilon_min,
                     intrinsic_coef, extrinsic_coef):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Q_ext = np.ones((state_size, action_size))
    Q_int = np.ones((state_size, action_size))
    true_counts = np.zeros(state_size, dtype=np.int32)

    total_timesteps = 0
    episode_reward = 0
    episode_length = 0
    sample_episode = 0
    intrinsic_rewards_over_time = []

    obs_rms.update(np.eye(state_size))
    state, _ = env.reset()
    true_counts[state] += 1

    for _ in range(5000):
        random_action = env.action_space.sample()
        next_state, _, terminated, truncated, _ = env.step(random_action)
        next_state_tensor = one_hot(next_state, state_size)
        obs_rms.update(next_state_tensor.numpy())

        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    state, _ = env.reset()
    state_tensor = one_hot(state, state_size).float()
    obs_rms.update(state_tensor.numpy())
    true_counts[state] += 1

    while total_timesteps < max_timesteps:
        Q_total = extrinsic_coef * Q_ext[state] + intrinsic_coef * Q_int[state]

        if np.random.rand() < epsilon or np.all(Q_total == 0):
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_total)

        next_state, ext_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        true_counts[next_state] += 1

        next_state_tensor = one_hot(next_state, state_size).float()
        normalized_obs = (next_state_tensor - torch.from_numpy(obs_rms.mean).float()) / torch.sqrt(
            torch.from_numpy(obs_rms.var).float() + 1e-8)
        with torch.no_grad():
            target = rnd.target(normalized_obs)
        pred = rnd.predictor(normalized_obs)
        int_reward = 0.5 * ((pred - target) ** 2).sum().item()

        discounted_int_reward = reward_filter.update(int_reward)
        reward_rms.update(np.array([discounted_int_reward]))
        normalized_int_reward = int_reward / np.sqrt(np.maximum(reward_rms.var, 1e-8))
        intrinsic_rewards_over_time.append(normalized_int_reward)

        total_reward = ext_reward + intrinsic_coef * normalized_int_reward

        next_Q_total = extrinsic_coef * Q_ext[next_state] + intrinsic_coef * Q_int[next_state]
        best_next_action = np.argmax(next_Q_total)

        Q_ext[state, action] += alpha * (
                ext_reward + gamma * Q_ext[next_state, best_next_action] - Q_ext[state, action]
        )

        Q_int[state, action] += alpha * (
                normalized_int_reward + gamma * Q_int[next_state, best_next_action] - Q_int[state, action]
        )

        forward_loss = torch.nn.functional.mse_loss(pred, target.detach())
        rnd_optimizer.zero_grad()
        forward_loss.backward()
        rnd_optimizer.step()

        wandb.log({
            "step": total_timesteps,
            "int_reward": intrinsic_coef * normalized_int_reward,
            "ext_reward": ext_reward,
            "Q_ext_max": Q_ext[state].max(),
            "Q_int_max": Q_int[state].max(),
            "forward_loss": forward_loss.item()
        }, step=total_timesteps)

        log_statewise_intrinsic_reward(state=next_state, int_reward=normalized_int_reward, step=total_timesteps)

        total_timesteps += 1
        episode_reward += ext_reward
        episode_length += 1

        if done:
            sample_episode += 1
            print(
                f"Timestep {total_timesteps}, Episode {sample_episode}, "
                f"Epsilon {epsilon:.3f}, Return {episode_reward:.2f}, "
                f"IntReward {intrinsic_coef * normalized_int_reward:.4f}, Length {episode_length}"
            )
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            state, _ = env.reset()
            state_tensor = one_hot(state, state_size).float()
            obs_rms.update(state_tensor.numpy())
            episode_reward = 0
            episode_length = 0
        else:
            state = next_state

    plot_q_heatmaps(Q_int, Q_ext, intrinsic_coef=intrinsic_coef)
    plot_intrinsic_reward_curve(intrinsic_rewards_over_time)

    return extrinsic_coef * Q_ext + intrinsic_coef * Q_int, true_counts


def log_statewise_intrinsic_reward(state, int_reward, step, prefix="int_reward"):
    wandb.log({f"{prefix}/state_{state}": int_reward}, step=step)


def plot_intrinsic_reward_curve(rewards, filename="intrinsic_reward_curve.png"):
    plt.figure()
    plt.plot(rewards)
    plt.title("Normalized Intrinsic Reward Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Intrinsic Reward")
    plt.savefig(filename)
    plt.close()


def plot_q_heatmaps(Q_int, Q_ext, intrinsic_coef=0.1, filename="q_heatmaps.png"):
    q_int_values = np.max(Q_int, axis=1)
    q_ext_values = np.max(Q_ext, axis=1)
    q_total_values = np.max(Q_ext + intrinsic_coef * Q_int, axis=1)

    side = int(np.sqrt(len(q_int_values)))
    assert side * side == len(q_int_values), "Q tables must correspond to square grid"

    q_int_grid = q_int_values.reshape(side, side)
    q_ext_grid = q_ext_values.reshape(side, side)
    q_total_grid = q_total_values.reshape(side, side)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axs[0].imshow(q_int_grid, cmap="viridis")
    axs[0].set_title("Q_int - Curiosity Map")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    fig.colorbar(im0, ax=axs[0], label="Max Q_int per state")

    im1 = axs[1].imshow(q_ext_grid, cmap="plasma")
    axs[1].set_title("Q_ext - Value Map")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    fig.colorbar(im1, ax=axs[1], label="Max Q_ext per state")

    im2 = axs[2].imshow(q_total_grid, cmap="inferno")
    axs[2].set_title("Q_total - Combined Map")
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Y")
    fig.colorbar(im2, ax=axs[2], label="Max Q_total per state")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def evaluate_agent(Q, env, episodes=100, gif_path="frozenlake_qlearning_20x20.gif"):
    total_rewards = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_rewards += episode_reward
    save_gif(Q, env, gif_path)
    return total_rewards / episodes


def save_gif(Q, env, filename="frozenlake_qlearning.gif", max_steps=700):
    frames = []
    state, _ = env.reset()
    filename = Path("frozen-lake-plots") / f"frozenlake_qlearning_rnd{int(np.sqrt(env.observation_space.n))}.gif"

    done = False
    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            frames.append(env.render())
            break
    imageio.mimsave(filename, frames, duration=0.3)
    print(f"GIF saved as {filename}")


def plot_intrinsic_vs_true_bonus_heatmap(
        rnd, true_counts, obs_rms, reward_rms, save_dir="frozen-lake-plots"
):
    state_size = len(true_counts)
    grid_size = int(np.sqrt(state_size))
    save_path = Path(save_dir) / f"frozenlake_rnd_vs_true_bonus_{grid_size}x{grid_size}.png"

    mean_tensor = torch.from_numpy(obs_rms.mean).float()
    var_tensor = torch.from_numpy(obs_rms.var).float()

    rnd_bonus = []
    for s in range(state_size):
        obs = one_hot(s, state_size)
        normalized_obs = (obs - mean_tensor) / torch.sqrt(var_tensor + 1e-8)
        with torch.no_grad():
            target = rnd.target(normalized_obs)
            pred = rnd.predictor(normalized_obs)
        bonus = 0.5 * ((pred - target) ** 2).sum().item()
        normalized_int_reward = bonus / np.sqrt(reward_rms.var)
        rnd_bonus.append(normalized_int_reward)
    rnd_bonus = np.array(rnd_bonus)

    true_bonus = 1.0 / np.sqrt(true_counts + 1e-8)
    visited_mask = (true_counts > 0)
    masked_true_bonus = true_bonus[visited_mask]
    masked_rnd_bonus = rnd_bonus[visited_mask]

    true_bonus_grid = np.ma.masked_where(~visited_mask, true_bonus).reshape(grid_size, grid_size)
    rnd_bonus_grid = np.ma.masked_where(~visited_mask, rnd_bonus).reshape(grid_size, grid_size)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    cmaps = ["magma", "magma"]
    titles = ["True Bonus", "Approx Bonus"]

    for ax, data, title, cmap in zip(axs[:2], [true_bonus_grid, rnd_bonus_grid], titles, cmaps):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))

    axs[2].scatter(masked_true_bonus, masked_rnd_bonus)
    axs[2].set_xlabel("True Bonus (1/sqrt(N))")
    axs[2].set_ylabel("RND Bonus (MSE)")
    axs[2].set_title("True vs. Approx Bonus")
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Heatmap and scatter plot saved to {save_path}")


def main():
    seed = 42
    set_seed(seed)

    wandb.init(project="frozenlake-cfn", name="true-vs-pseudo-counts")
    map = [
        "SHFFFFFF",
        "FFFFFHFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFHFFFFF",
        "FFFFFFFF",
        "FFFFFHFF",
        "FFFHFFFG",
    ]

    # map = [
    #     "SFFFFFFF",
    #     "FFFFFFFF",
    #     "FFFFFFFF",
    #     "FFFFFFFF",
    #     "FFFFFFFF",
    #     "FFFFFFFF",
    #     "FFFFFFFF",
    #     "FFFFFFFG",
    # ]

    # map = [
    #     "SFFFFFFFFFFF",
    #     "FFFFFFFFFFFF",
    #     "FFFFFFFFFFFF",
    #     "FFFFFFFFFFFF",
    #     "FFFFFFFFFFFF",
    #     "FFFFFFFFFFFF",
    #     "FFFFFFFFFFFF",
    #     "FFFFFFFFFFFF",
    #     "FFFFFFFFFFFF",
    #     "FFFFFFFFFFFF",
    #     "FFFFFFFFFFFF",
    #     "FFFFFFFFFFFG"
    # ]

    # TODO: Try this on the KISLURM
    # map = [
    #     "SFFFFFHFFFFHFHFH",
    #     "FHFFFFHFFFFFHFFF",
    #     "FFFHFFFFHFFFFHFF",
    #     "FHHFFHHFFFHFHFFF",
    #     "FFHFFFFFFHFFFHFF",
    #     "FFFFFHFFFHFHHFFF",
    #     "FHFHFFFFFHFFFFHF",
    #     "FFFHFHFFFFFHFHFF",
    #     "FHFFFFFHFFFFFHFF",
    #     "FFFFFHFFHHFFFFHF",
    #     "FHFFFHFFFHHFFFFF",
    #     "FFFFFHFFFFFHFFFF",
    #     "FFHFHFHFFHFHFHFF",
    #     "FFHFFFFFHFFFFFHF",
    #     "FFFFHFHFFFFHFFFF",
    #     "FFFFFFHFHFHFFFGF"
    # ]

    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array", desc=map, max_episode_steps=200)
    state_size = env.observation_space.n
    rnd = RNDModel(env.observation_space.n)
    rnd_optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-4)
    obs_rms = RunningMeanStd(shape=(state_size,))
    reward_rms = RunningMeanStd()
    discounted_reward = RewardForwardFilter(0.99)
    intrinsic_coef = 0.01

    Q, true_counts = train_q_learning(
        env,
        rnd,
        rnd_optimizer,
        obs_rms,
        reward_rms,
        discounted_reward,
        max_timesteps=50000,
        intrinsic_coef=intrinsic_coef,
        extrinsic_coef=2.0,
        alpha=1.0,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.1
    )

    print(true_counts.reshape(int(np.sqrt(len(true_counts))), int(np.sqrt(len(true_counts)))))

    plot_intrinsic_vs_true_bonus_heatmap(rnd, true_counts, obs_rms, reward_rms)

    avg_reward = evaluate_agent(Q, env)
    print(f"\nAverage evaluation reward over 100 episodes: {avg_reward:.2f}")


if __name__ == "__main__":
    main()
