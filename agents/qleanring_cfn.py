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


def update_cfn_network(cfn, optimizer, obs_batch, coin_flip_batch):
    predicted = cfn(obs_batch)
    loss = torch.nn.functional.mse_loss(predicted, coin_flip_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_q_learning(env, max_timesteps, alpha, gamma, epsilon, epsilon_decay, epsilon_min,
                     cfn, cfn_buffer, cfn_optimizer, coin_flip_dim):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Q = np.zeros((state_size, action_size))
    true_counts = np.zeros(state_size, dtype=np.int32)

    total_timesteps = 0
    episode_reward = 0
    episode_length = 0

    state, _ = env.reset()
    obs_tensor = one_hot(state, state_size)
    true_counts[state] += 1

    while total_timesteps < max_timesteps:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            if np.all(Q[state] == 0):
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # if done and next_state == 63:
        #     reward = 10.0

        intrinsic_reward = compute_intrinsic_reward(
            coin_flip_dim,
            cfn.compute_squared_output_norm(obs_tensor)
        )

        total_reward = reward + intrinsic_reward.item()

        _ = cfn(obs_tensor, update_prior_stats=True)

        coin_flip = get_coin_flips(coin_flip_dim)
        cfn_buffer.add(
            obs=obs_tensor.cpu().numpy().squeeze(),
            coin_flip=coin_flip.detach().cpu().numpy(),
            priority=1.0
        )

        best_next_action = np.argmax(Q[next_state])

        if done:
            target = reward + intrinsic_reward.item()   # no bootstrap at terminal
        else:
            target = reward + intrinsic_reward.item() + gamma * Q[next_state, best_next_action]

        Q[state, action] += alpha * (target - Q[state, action])


        obs_batch_bc, coin_flip_batch_bc, indices = cfn_buffer.sample_with_indices(batch_size=1024)
        update_cfn_network(cfn, cfn_optimizer, obs_batch_bc, coin_flip_batch_bc)
        cfn_buffer.update_priorities(indices, obs_batch_bc, cfn, coin_flip_dim)

        if total_timesteps % 1000 == 0 and total_timesteps > 0:
            avg_reward = evaluate_agent(Q, env, episodes=20)
            wandb.log({"eval/avg_reward": avg_reward}, step=total_timesteps)

        total_timesteps += 1
        episode_reward += reward
        episode_length += 1
        state = next_state
        obs_tensor = one_hot(state, state_size)
        true_counts[state] += 1
        _ = cfn(obs_tensor, update_prior_stats=True)

        if done:
            print(
                f"Timestep {total_timesteps}, "
                f"Epsilon {epsilon:.3f}, Return {episode_reward:.2f}, "
                f"Length {episode_length}"
            )

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            state, _ = env.reset()
            obs_tensor = one_hot(state, state_size)
            true_counts[state] += 1
            episode_reward = 0
            episode_length = 0

    return Q, true_counts


def evaluate_agent(Q, env, episodes=100, save_gif_at_end=False, gif_path="frozenlake_qlearning_20x20.gif"):
    total_rewards = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_rewards += reward

    if save_gif_at_end:
        save_gif(Q, env, gif_path)
    return total_rewards / episodes


def save_gif(Q, env, filename="frozenlake_qlearning.gif", max_steps=100):
    frames = []
    state, _ = env.reset()
    filename = Path("frozen-lake-plots") / f"frozenlake_qlearning{int(np.sqrt(env.observation_space.n))}.gif"

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
    axs[2].plot([0, max_val], [0, max_val], linestyle="--", color="black", linewidth=1)
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
    vmins = [count_vmin, count_vmin, bonus_vmin, bonus_vmin]
    vmaxs = [count_vmax, count_vmax, bonus_vmax, bonus_vmax]

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


def train_q_learning_vanilla(env, max_timesteps, alpha, gamma, epsilon, epsilon_decay, epsilon_min,
                             buffer_size=50000, batch_size=64):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Q = np.zeros((state_size, action_size))
    true_counts = np.zeros(state_size, dtype=np.int32)

    replay_buffer = deque(maxlen=buffer_size)

    total_timesteps = 0
    episode_reward = 0
    episode_length = 0
    state, _ = env.reset()
    true_counts[state] += 1

    while total_timesteps < max_timesteps:
        if np.random.rand() < epsilon or np.all(Q[state] == 0):
            action = np.random.choice(action_size)
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done and next_state == 63:
            reward = 10.0

        replay_buffer.append((state, action, reward, next_state, done))

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            for s, a, r, s_next, d in batch:
                target = r + (0.0 if d else gamma * np.max(Q[s_next]))
                Q[s, a] += alpha * (target - Q[s, a])

        if total_timesteps % 1000 == 0 and total_timesteps > 0:
            avg_reward = evaluate_agent(Q, env, episodes=20)
            wandb.log({"eval/avg_reward_van": avg_reward, "eval/step_van": total_timesteps})

        total_timesteps += 1
        episode_reward += reward
        episode_length += 1
        state = next_state
        true_counts[state] += 1

        if done:
            print(
                f"[Vanilla Q] Timestep {total_timesteps}, Return {episode_reward:.2f}, Length {episode_length}"
            )
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            state, _ = env.reset()
            true_counts[state] += 1
            episode_reward = 0
            episode_length = 0

    return Q, true_counts


def q_learning_cfn_main(cfg):

    map = cfg.agent.env.desc
    # map = [
    #     "SFFFFFFH",
    #     "HHHHFFFH",
    #     "FFFFFHFF",
    #     "FGFFFFFH",
    #     "FHFFFHFF",
    #     "FHFFFFHF",
    #     "FFFFHHHF",
    #     "HHHHHFFG",
    # ]
    #
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
    #
    # map = [
    #     "SFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFF",
    #     "FFFFFFFFFFFFFFFG"
    # ]

    env = gym.make(cfg.agent.env.id, is_slippery=cfg.agent.env.is_slippery, render_mode=cfg.agent.env.render_mode, desc=map, max_episode_steps=cfg.agent.env.max_episode_steps)
    state_size = env.observation_space.n
    coin_flip_dim = cfg.agent.q_learning_cfn.intrinsic.coin_flip_dim

    cfn = CoinFlipNetwork(state_dim=state_size, coin_dim=coin_flip_dim)
    cfn_buffer = CFNReplayBufferWrapper(
        size=cfg.agent.cfn_config.buffer.size,
        obs_shape=(state_size,),
        coin_flip_dim=coin_flip_dim,
        alpha=cfg.agent.cfn_config.buffer.alpha,
    )
    cfn_optimizer = optim.Adam(cfn.parameters(), lr=cfg.agent.cfn_config.lr)

    Q, true_counts = train_q_learning(
        env=env,
        max_timesteps=cfg.agent.q_learning_cfn.max_timesteps,
        alpha=cfg.agent.q_learning_cfn.alpha,
        gamma=cfg.agent.q_learning_cfn.gamma,
        epsilon=cfg.agent.q_learning_cfn.epsilon.start,
        epsilon_decay=cfg.agent.q_learning_cfn.epsilon.decay,
        epsilon_min=cfg.agent.q_learning_cfn.epsilon.min,
        cfn=cfn,
        cfn_buffer=cfn_buffer,
        cfn_optimizer=cfn_optimizer,
        coin_flip_dim=coin_flip_dim
    )

    Q_vanilla, true_counts_vanilla = train_q_learning_vanilla(
        env=env,
        max_timesteps=cfg.agent.q_learning_vanilla.max_timesteps,
        alpha=cfg.agent.q_learning_vanilla.alpha,
        gamma=cfg.agent.q_learning_vanilla.gamma,
        epsilon=cfg.agent.q_learning_vanilla.epsilon.start,
        epsilon_decay=cfg.agent.q_learning_vanilla.epsilon.decay,
        epsilon_min=cfg.agent.q_learning_vanilla.epsilon.min,
        buffer_size=cfg.agent.q_learning_vanilla.replay.buffer_size,
        batch_size=cfg.agent.q_learning_vanilla.replay.batch_size
    )

    avg_reward = evaluate_agent(Q, env, save_gif_at_end=True)
    print(f"\nAverage evaluation reward over 100 episodes: {avg_reward:.2f}")

    plot_combined_bonus_comparison(cfn, true_counts, coin_flip_dim)

    print(true_counts.reshape(int(np.sqrt(len(true_counts))), int(np.sqrt(len(true_counts)))))
    print(true_counts_vanilla.reshape(int(np.sqrt(len(true_counts_vanilla))), int(np.sqrt(len(true_counts_vanilla)))))


if __name__ == "__main__":
    q_learning_cfn_main()
