from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import imageio
import gymnasium as gym
from matplotlib import pyplot as plt

import wandb

from CFN.CFN import CoinFlipNetwork
from CFN.cfn_buffer import CFNReplayBufferWrapper
from CFN.priority_util import compute_intrinsic_reward, get_coin_flips
import random

from utils.env_wrapper import ActionConfusionWrapper


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


def update_cfn(cfn, optimizer, obs_batch, coin_flip_batch):
    predicted = cfn(obs_batch)
    loss = torch.nn.functional.mse_loss(predicted, coin_flip_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_q_learning(
    env,
    max_timesteps,
    alpha,
    gamma,
    epsilon,
    epsilon_decay,
    epsilon_min,
    cfn,
    cfn_buffer,
    cfn_optimizer,
    coin_flip_dim,
    buffer_size,
):


    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA))
    buffer = deque(maxlen=50000)

    step = 0
    episode = 0
    goals = 0
    first_goal_step = None
    max_depth = []

    state, _ = env.reset()
    obs = one_hot(state, nS)

    while step < max_timesteps:
        # ε-greedy ONLY (no CFN bias here)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done and next_state == nS - 1:
            reward = 10.0
            goals += 1
            if first_goal_step is None:
                first_goal_step = step
            print("🎯 Goal reached")

        # intrinsic reward (decays to zero)
        beta = max(0.0, 1.0 - step / (0.5 * max_timesteps))
        intrinsic = beta * 0.05 * compute_intrinsic_reward(
            coin_flip_dim,
            cfn.compute_squared_output_norm(obs)
        ).item()

        buffer.append((state, action, reward, intrinsic, next_state, done))

        # CFN update
        coins = get_coin_flips(coin_flip_dim)
        cfn_buffer.add(
            obs=obs.numpy().squeeze(),
            coin_flip=coins.detach().numpy(),
            priority=1.0
        )

        obs_b, coin_b, idx = cfn_buffer.sample_with_indices(512)
        cfn_loss = update_cfn(cfn, cfn_optimizer, obs_b, coin_b)
        cfn_buffer.update_priorities(idx, obs_b, cfn, coin_flip_dim)

        # Q update
        if len(buffer) >= 256:
            batch = random.sample(buffer, 256)
            for s, a, r, ir, sn, d in batch:
                if d:
                    target = r
                else:
                    target = r + ir + gamma * np.max(Q[sn])
                Q[s, a] += alpha * (target - Q[s, a])

        state = next_state
        obs = one_hot(state, nS)
        step += 1

        if done:
            episode += 1
            max_depth.append(state // int(np.sqrt(nS)))
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            state, _ = env.reset()
            obs = one_hot(state, nS)

        if step % 5000 == 0:
            wandb.log({
                "steps": step,
                "epsilon": epsilon,
                "beta": beta,
                "cfn_loss": cfn_loss,
                "goal_reaches": goals,
                "mean_max_depth": np.mean(max_depth[-50:]) if max_depth else 0
            })

    return Q, goals, first_goal_step, max_depth


def evaluate_agent(
    Q,
    env,
    episodes=100,
    save_gif_at_end=False,
    gif_path="frozenlake_qlearning.gif",
):
    total_return = 0.0
    success_episodes = 0

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_return = 0.0
        reached_goal = False

        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done and state == env.observation_space.n - 1:
                reward = 1
                reached_goal = True
            else:
                reward = 0

            episode_return += reward

        total_return += episode_return
        if reached_goal:
            success_episodes += 1

    avg_return = total_return / episodes
    success_rate = success_episodes / episodes

    print(f"Goal reached in {success_episodes}/{episodes} episodes")
    print(f"Average return: {avg_return:.2f}")

    if save_gif_at_end:
        save_gif(Q, env, gif_path)

    return avg_return, success_rate



def save_gif(Q, env, filename=None, max_steps=100):
    frames = []
    state, _ = env.reset()
    filename = Path("frozen-lake-plots") / f"frozenlake_qlearning{int(np.sqrt(env.observation_space.n))}_{filename}.gif"

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


def plot_combined_bonus_comparison(cfn, true_counts, coin_flip_dim, save_path="combined_bonus_plot.png"):
    state_size = len(true_counts)
    grid_size = int(np.sqrt(state_size))
    save_path = Path("frozen-lake-plots") / f"frozenlake_qlearning{int(grid_size)}.png"

    states = list(range(state_size))

    pseudo_count = np.array([
        coin_flip_dim / cfn.compute_squared_output_norm(one_hot(s, state_size)).item()
        for s in states
    ])

    true_bonus = np.array([1 / np.sqrt(c + 1e-8) for c in true_counts])
    approx_bonus = np.array([
        np.sqrt(cfn.compute_squared_output_norm(one_hot(s, state_size)).item() / coin_flip_dim)
        for s in states
    ])

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    max_val = max(max(true_counts), max(pseudo_count))

    axs[0].scatter(true_counts, pseudo_count)
    axs[0].plot([0, max_val], [0, max_val], linestyle="--", color="black", linewidth=1)
    axs[0].set_xlabel("True count")
    axs[0].set_ylabel("Pseudo Count")
    axs[0].grid(True)

    axs[1].plot(states, true_counts, label="True count", marker='o')
    axs[1].plot(states, pseudo_count, label="Pseudo count", marker='x')
    axs[1].set_xlabel("State")
    axs[1].set_ylabel("Counts")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].scatter(true_bonus, approx_bonus)
    axs[2].plot([0, 0.2], [0, 0.2], linestyle="--", color="black", linewidth=1)
    axs[2].set_xlabel("True Bonus")
    axs[2].set_ylabel("Approx Bonus")
    axs[2].set_title("True vs. Approx Bonus")
    axs[2].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Combined plot saved to {save_path}")

    true_counts_grid = np.array(true_counts).reshape(grid_size, grid_size)
    pseudo_counts_grid = np.array(pseudo_count).reshape(grid_size, grid_size)
    true_bonus_grid = np.array(true_bonus).reshape(grid_size, grid_size)
    approx_bonus_grid = np.array(approx_bonus).reshape(grid_size, grid_size)

    count_vmin = min(true_counts_grid.min(), pseudo_counts_grid.min())
    count_vmax = max(true_counts_grid.max(), pseudo_counts_grid.max())
    bonus_vmin = min(true_bonus_grid.min(), approx_bonus_grid.min())
    bonus_vmax = max(true_bonus_grid.max(), approx_bonus_grid.max())

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    titles = [
        "True Counts", "Pseudo Counts",
        "True Bonus", "Approx Bonus"
    ]
    data_grids = [
        true_counts_grid, pseudo_counts_grid,
        true_bonus_grid, approx_bonus_grid
    ]
    cmaps = ["viridis", "viridis", "magma", "magma"]

    axs = axs.flatten()

    for ax, data, title, cmap in zip(axs, data_grids, titles, cmaps):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_xticklabels(range(grid_size))
        ax.set_yticklabels(range(grid_size))

    plt.tight_layout()
    heatmap_path = save_path.with_name(save_path.stem + "_heatmap.png")
    plt.savefig(heatmap_path)
    print(f"Heatmap plot saved to {heatmap_path}")
    plt.close()


def log_counts_to_wandb(cfn, true_counts, coin_flip_dim):
    state_size = len(true_counts)
    states = list(range(state_size))

    true_bonus = [1 / (true_count + 1e-8) for true_count in true_counts]
    pseudo_bonus = [
        np.sqrt(cfn.compute_squared_output_norm(one_hot(s, state_size)).item() / coin_flip_dim)
        for s in states
    ]

    for s in states:
        wandb.log({
            "true_bonus": true_bonus[s],
            "pseudo_bonus": pseudo_bonus[s],
        })

    print("\nState\tTrue Count\tPseudo Bonus (sqrt(norm² / d))")
    for s in states:
        print(f"{s}\t{true_counts[s]}\t\t{pseudo_bonus[s]:.4f}")


def train_q_learning_vanilla(
    env,
    max_timesteps,
    alpha,
    gamma,
    epsilon,
    epsilon_decay,
    epsilon_min,
    buffer_size=50000,
    batch_size=64,
):
    nS = env.observation_space.n
    nA = env.action_space.n

    Q = np.zeros((nS, nA))
    replay_buffer = deque(maxlen=buffer_size)

    total_steps = 0
    episode_reward = 0
    episode_length = 0

    goals = 0
    first_goal_step = None
    max_depth_per_episode = []

    state, _ = env.reset()

    while total_steps < max_timesteps:
        if np.random.rand() < epsilon or np.all(Q[state] == 0):
            action = np.random.choice(nA)
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done and next_state == nS - 1:
            reward = 10.0
            goals += 1
            if first_goal_step is None:
                first_goal_step = total_steps
            print("🎯 [Vanilla] Goal reached")

        replay_buffer.append((state, action, reward, next_state, done))

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            for s, a, r, s_next, d in batch:
                target = r if d else r + gamma * np.max(Q[s_next])
                Q[s, a] += alpha * (target - Q[s, a])

        total_steps += 1
        episode_reward += reward
        episode_length += 1
        state = next_state

        if done:
            max_depth_per_episode.append(state // int(np.sqrt(nS)))

            print(
                f"[Vanilla] Step {total_steps}, "
                f"Return {episode_reward:.2f}, "
                f"Length {episode_length}, "
                f"Goals {goals}"
            )

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0

        if total_steps % 5000 == 0:
            wandb.log({
                "vanilla/steps": total_steps,
                "vanilla/epsilon": epsilon,
                "vanilla/goals": goals,
                "vanilla/first_goal_step": first_goal_step if first_goal_step else -1,
                "vanilla/mean_max_depth": np.mean(max_depth_per_episode[-50:]) if max_depth_per_episode else 0,
            })

    return Q, goals, first_goal_step, max_depth_per_episode


def main():
    # seed = 42
    # set_seed(seed)

    wandb.init(project="frozenlake-cfn", name="true-vs-pseudo-counts-aggressive")

    map = [
        "SFFFFFFH",
        "HHHHFFFH",
        "FFFFFHFF",
        "FGFFFFFH",
        "FHFFFHFF",
        "FHFFFFHF",
        "FFFFHHHF",
        "HHHHHFFG",
    ]

    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array", desc=map, max_episode_steps=500)

    state_size = env.observation_space.n
    coin_flip_dim = 16

    cfn = CoinFlipNetwork(state_dim=state_size, coin_dim=coin_flip_dim)
    cfn_buffer = CFNReplayBufferWrapper(
        size=100000,
        obs_shape=(state_size,),
        coin_flip_dim=coin_flip_dim,
        alpha=0.5
    )
    cfn_optimizer = optim.Adam(cfn.parameters(), lr=1e-3)

    # More aggressive hyperparameters
    Q, goals, first_goal_step, max_depth = train_q_learning(
        env=env,
        max_timesteps=300000,      # Even more timesteps
        alpha=0.15,                # Higher learning rate
        gamma=0.997,               # Higher discount for long horizon
        epsilon=1.0,
        epsilon_decay=0.99975,     # Even slower decay
        epsilon_min=0.1,           # Higher minimum exploration
        cfn=cfn,
        cfn_buffer=cfn_buffer,
        cfn_optimizer=cfn_optimizer,
        coin_flip_dim=coin_flip_dim,
        buffer_size=50000
    )

    wandb.log({
        "cfn/train_goals": goals,
        "cfn/first_goal_step": first_goal_step if first_goal_step is not None else -1,
        "cfn/mean_max_depth": np.mean(max_depth),
        "cfn/max_depth": np.max(max_depth),
    })



    Q_vanilla, goals_vanilla, first_goal_vanilla, depth_vanilla = train_q_learning_vanilla(
        env=env,
        max_timesteps=300000,
        alpha=0.1,
        gamma=0.999,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.05,
        buffer_size=50000,
        batch_size=64
    )

    wandb.log({
        "vanilla/train_goals": goals_vanilla,
        "vanilla/first_goal_step": first_goal_vanilla if first_goal_vanilla is not None else -1,
        "vanilla/mean_max_depth": np.mean(depth_vanilla),
        "vanilla/max_depth": np.max(depth_vanilla),
    })


    avg_ret_cfn, success_cfn = evaluate_agent(Q, env, episodes=100, save_gif_at_end=True, gif_path="cfn")
    avg_ret_van, success_van = evaluate_agent(Q_vanilla, env, episodes=100, save_gif_at_end=True, gif_path="vanilla")

    wandb.log({
        "cfn/eval_avg_return": avg_ret_cfn,
        "cfn/eval_success_rate": success_cfn,
        "vanilla/eval_avg_return": avg_ret_van,
        "vanilla/eval_success_rate": success_van,
    })



if __name__ == "__main__":
    main()