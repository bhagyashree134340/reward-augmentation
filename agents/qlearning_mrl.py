import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import imageio
import gymnasium as gym
from cpprb import ReplayBuffer
from matplotlib import pyplot as plt
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_true_vs_munchausen_bonus_heatmap(Q, true_counts, tau, alpha_m, lo, env, save_dir="frozen-lake-plots",
                                          plotname="true_vs_mrl_bonus"):
    state_size = len(true_counts)
    grid_size = int(np.sqrt(state_size))
    save_path = Path(save_dir) / f"{plotname}_{grid_size}x{grid_size}.png"

    true_bonus = 1.0 / np.sqrt(true_counts + 1e-8)

    munchausen_bonus = np.zeros(state_size)
    for s in range(state_size):
        q_vals = Q[s]
        q_tensor = torch.tensor(q_vals, dtype=torch.float32, device=device)
        probs = F.softmax(q_tensor / tau, dim=0).cpu().numpy()
        a_star = np.argmax(q_vals)
        log_pi = np.log(np.clip(probs[a_star], 1e-8, 1.0))
        log_pi = np.clip(log_pi, lo, 0.0)
        munchausen_bonus[s] = alpha_m * tau * log_pi

    visited_mask = (true_counts > 0)
    masked_true_bonus_grid = np.ma.masked_where(~visited_mask, true_bonus).reshape(grid_size, grid_size)
    mrl_bonus_grid = munchausen_bonus.reshape(grid_size, grid_size)

    masked_true_bonus = true_bonus[visited_mask]
    masked_mrl_bonus = munchausen_bonus[visited_mask]
    outlier_mask = masked_true_bonus < 1e3

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    cmaps = ["magma", "magma"]
    titles = ["True Bonus", "Munchausen Bonus"]

    for ax, data, title, cmap in zip(axs[:2], [masked_true_bonus_grid, mrl_bonus_grid], titles, cmaps):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_xticklabels(range(grid_size))
        ax.set_yticklabels(range(grid_size))

    axs[2].scatter(masked_true_bonus[outlier_mask], masked_mrl_bonus[outlier_mask])
    axs[2].set_xlabel("True Bonus")
    axs[2].set_ylabel("Munchausen Bonus")
    axs[2].set_title("True vs. Munchausen Bonus")
    axs[2].grid(True)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Munchausen vs True bonus heatmap and scatter saved to: {save_path}")


def train_q_learning(env, max_timesteps, alpha, tau, gamma, epsilon, epsilon_decay, epsilon_min, alpha_m, lo,
                     buffer_size=50000, batch_size=64):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Q = np.zeros((state_size, action_size))
    true_counts = np.zeros(state_size, dtype=np.int32)

    def compute_log_pi(q_values, tau):
        v = np.max(q_values)
        logsumexp = np.log(np.sum(np.exp((q_values - v) / tau))) + v / tau
        log_pi = (q_values / tau) - logsumexp
        return log_pi

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

        replay_buffer.append((state, action, reward, next_state, done))
        true_counts[next_state] += 1

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            for s, a, r, ns, d in batch:
                log_pi = compute_log_pi(Q[s], tau)
                log_pi_a = np.clip(log_pi[a], lo, 0.0)
                munchausen_reward = r + alpha_m * log_pi_a

                next_log_pi = compute_log_pi(Q[ns], tau)
                pi_next = np.exp(next_log_pi)
                soft_v_next = np.sum(pi_next * (Q[ns] - tau * next_log_pi))

                target = munchausen_reward + (0.0 if d else gamma * soft_v_next)
                Q[s, a] += alpha * (target - Q[s, a])

        state = next_state
        total_timesteps += 1
        episode_reward += reward
        episode_length += 1

        if done:
            print(
                f"[MRL] Timestep {total_timesteps}, Return {episode_reward:.2f}, Length {episode_length}"
            )
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            state, _ = env.reset()
            true_counts[state] += 1
            episode_reward = 0
            episode_length = 0

    return Q, true_counts


def train_q_learning_without_mrl(env, max_timesteps, alpha, gamma, epsilon, epsilon_decay, epsilon_min,
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

        replay_buffer.append((state, action, reward, next_state, done))

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            for s, a, r, s_next, d in batch:
                target = r + (0.0 if d else gamma * np.max(Q[s_next]))
                Q[s, a] += alpha * (target - Q[s, a])

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


def softmax_probs(q_vals, entropy_tau):
    q_tensor = torch.tensor(q_vals, dtype=torch.float32, device=device)
    probs_tensor = F.softmax(q_tensor / entropy_tau, dim=0)
    return probs_tensor.cpu().numpy()


def evaluate_agent(Q, env, episodes=100, gif_path="frozenlake_qlearning_20x20.gif", tau=0.03):
    total_rewards = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action_probs = softmax_probs(Q[state], tau)
            action = np.random.choice(len(action_probs), p=action_probs)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_rewards += reward
    save_gif(Q, env, filename=gif_path, tau=tau)
    return total_rewards / episodes


def save_gif(Q, env, filename="frozenlake_qlearning.gif", max_steps=100, tau=0.03):
    frames = []
    state, _ = env.reset()
    filename = Path("frozen-lake-plots") / f"frozenlake_qlearning{int(np.sqrt(env.observation_space.n))}.gif"

    for _ in range(max_steps):
        frame = env.render()
        frames.append(frame)
        action_probs = softmax_probs(Q[state], tau)
        action = np.random.choice(len(action_probs), p=action_probs)
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            frames.append(env.render())
            break
    imageio.mimsave(filename, frames, duration=0.3)
    print(f"GIF saved as {filename}")


def compute_avg_gap(Q):
    return np.mean(np.max(Q, axis=1) - np.partition(Q, -2, axis=1)[:, -2])


def print_policy(Q, env, tau=0.03, action_symbols=None):
    nrow = env.unwrapped.nrow
    ncol = env.unwrapped.ncol
    num_actions = Q.shape[1]

    print("\nSoftmax Policy π(a|s) for each state (greedy action shown):\n")

    for i in range(nrow):
        row = ""
        for j in range(ncol):
            state = i * ncol + j
            probs = softmax_probs(Q[state], tau)
            best_action = np.argmax(probs)
            if action_symbols:
                action_str = action_symbols[best_action]
            else:
                action_str = str(best_action)
            row += f"{action_str}({probs[best_action]:.2f})\t"
        print(row)


def compute_avg_action_gap(Q):
    sorted_q = np.sort(Q, axis=1)
    max_q = sorted_q[:, -1]
    second_best_q = sorted_q[:, -2]
    gap = max_q - second_best_q
    return np.mean(gap)


def plot_action_gap_lines(Q_mrl, Q_vanilla, env, plotname="action_gap_lineplot"):
    num_states = Q_mrl.shape[0]

    def compute_gaps(Q):
        sorted_q = np.sort(Q, axis=1)
        return sorted_q[:, -1] - sorted_q[:, -2]

    gaps_mrl = compute_gaps(Q_mrl)
    gaps_vanilla = compute_gaps(Q_vanilla)

    states = np.arange(num_states)

    plt.figure(figsize=(10, 5))
    plt.plot(states, gaps_mrl, label="Munchausen", linewidth=2)
    plt.plot(states, gaps_vanilla, label="Vanilla", linewidth=2)
    plt.xlabel("State Index")
    plt.ylabel("Action Gap (Q_max - Q_2nd)")
    plt.title("Per-State Action Gaps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = Path("frozen-lake-plots") / f"{plotname}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"Action gap line plot saved to: {save_path}")
    plt.close()


def plot_action_gap_heatmap(Q, env, plotname="action_gap_heatmap"):
    nrow = env.unwrapped.nrow
    ncol = env.unwrapped.ncol
    num_states = Q.shape[0]

    action_gaps = np.zeros(num_states)
    for s in range(num_states):
        q_vals = Q[s]
        sorted_q = np.sort(q_vals)
        action_gaps[s] = sorted_q[-1] - sorted_q[-2]  # max - second-best

    heatmap = action_gaps.reshape(nrow, ncol)

    plt.figure(figsize=(8, 6))
    plt.title("Action Gap per State")
    plt.imshow(heatmap, cmap="plasma", origin="upper")
    plt.colorbar(label="Q(s, a*) - Q(s, a_2nd_best)")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()

    save_path = Path("frozen-lake-plots") / f"{plotname}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"Action gap heatmap saved to: {save_path}")
    plt.close()


def main():
    # seed = 42
    # set_seed(seed)

    wandb.init(project="frozenlake-cfn", name="MRL-true-vs-pseudo-bonues", mode="disabled")

    map = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFG",
    ]

    # map = [
    #     "SFFFF",
    #     "FFFFF",
    #     "FFFFF",
    #     "FFFFF",
    #     "FFFFG"
    # ]

    # map = [
    #     "SHFFFFFF",
    #     "FFFFFHFF",
    #     "FFFFFFFF",
    #     "FFFFFFFF",
    #     "FFHFFFFF",
    #     "FFFFFFFF",
    #     "FFFFFHFF",
    #     "FFFHFFFG",
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

    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array", desc=map, max_episode_steps=200)

    Q_mrl, true_counts = train_q_learning(
        env=env,
        batch_size=128,
        max_timesteps=50000,
        alpha=1.0,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.05,
        tau=0.03,
        alpha_m=0.9,
        lo=-1.0
    )

    Q_vanilla, true_counts_vanilla = train_q_learning_without_mrl(
        env=env,
        max_timesteps=50000,
        alpha=1.0,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.05
    )

    gap_mrl = compute_avg_action_gap(Q_mrl)
    gap_vanilla = compute_avg_action_gap(Q_vanilla)

    print(f"\nAverage Action Gap with Munchausen: {gap_mrl:.4f}")
    print(f"Average Action Gap without Munchausen: {gap_vanilla:.4f}")

    plot_action_gap_heatmap(Q_mrl, env, plotname="action_gap_heatmap_mrl")
    plot_action_gap_heatmap(Q_vanilla, env, plotname="action_gap_heatmap_vanilla")

    plot_action_gap_lines(Q_mrl, Q_vanilla, env, plotname="action_gap_lineplot_mrl_vs_vanilla")

    print("Q-values at state 0:")
    for a in range(env.action_space.n):
        print(f"Action {a}: {Q_mrl[0, a]:.4f}")

    avg_reward = evaluate_agent(Q_mrl, env)
    print(f"\nAverage evaluation reward over 100 episodes: {avg_reward:.2f}")

    plot_true_vs_munchausen_bonus_heatmap(
        Q=Q_mrl,
        true_counts=true_counts,
        tau=0.03,
        alpha_m=0.9,
        lo=-1.0,
        env=env,
        plotname="munchausen_vs_true_bonus"
    )
    # TODO: add heatmap for Q_vaniila
    # plot_munchausen_heatmap(Q_mrl, env, tau=0.05, alpha_m=0.3, lo=-0.1, plotname="munchausen_heatmap_run1")

    print(true_counts.reshape(int(np.sqrt(len(true_counts))), int(np.sqrt(len(true_counts)))))

    # action_symbols = {
    #     0: '←',
    #     1: '↓',
    #     2: '→',
    #     3: '↑',
    # }
    #
    # print_policy(Q, env, tau=0.03, action_symbols=action_symbols)


if __name__ == "__main__":
    main()
