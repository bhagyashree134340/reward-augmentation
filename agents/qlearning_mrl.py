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
                     buffer_size=50000, batch_size=64, noise_std=0.5):
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
        # reward += np.random.normal(loc=0.0, scale=noise_std)
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

        if total_timesteps % 1000 == 0:
            avg_reward = evaluate_agent(Q, env)
            wandb.log({"avg_return_mrl": avg_reward}, step=total_timesteps)

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
                                 buffer_size=50000, batch_size=64, noise_std=0.5):
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
        # reward += np.random.normal(loc=0.0, scale=noise_std)
        done = terminated or truncated

        replay_buffer.append((state, action, reward, next_state, done))

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            for s, a, r, s_next, d in batch:
                target = r + (0.0 if d else gamma * np.max(Q[s_next]))
                Q[s, a] += alpha * (target - Q[s, a])

        if total_timesteps % 1000 == 0:
            avg_reward = evaluate_agent(Q, env)
            wandb.log({"avg_return_vanilla": avg_reward, "timestep_vanilla": total_timesteps})

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


def evaluate_agent(Q, env, episodes=100, is_save_gif=False, gif_path="frozenlake_qlearning.gif", step=None,
                   prefix="eval"):
    total_rewards = 0
    for i in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = int(np.argmax(Q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_rewards += episode_reward

        if step is not None:
            wandb.log({f"{prefix}/reward": episode_reward}, step=step + i)

    avg_return = total_rewards / episodes

    if is_save_gif:
        save_gif(Q, env, filename=gif_path)

    return avg_return


def save_gif(Q, env, filename="frozenlake_qlearning.gif", max_steps=100, tau=0.03):
    frames = []
    state, _ = env.reset()
    filename = Path("frozen-lake-plots") / f"{filename.rstrip('.gif')}_{int(np.sqrt(env.observation_space.n))}x{int(np.sqrt(env.observation_space.n))}.gif"

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


def print_policy(Q, env, action_symbols=None):
    nrow = env.unwrapped.nrow
    ncol = env.unwrapped.ncol

    print("\nGreedy Policy π(a|s) for each state:\n")

    for i in range(nrow):
        row = ""
        for j in range(ncol):
            state = i * ncol + j
            best_action = np.argmax(Q[state])
            if action_symbols:
                action_str = action_symbols[best_action]
            else:
                action_str = str(best_action)
            row += f"{action_str}\t"
        print(row)


def compute_avg_action_gap(Q):
    sorted_q = np.sort(Q, axis=1)
    max_q = sorted_q[:, -1]
    second_best_q = sorted_q[:, -2]
    gap = max_q - second_best_q
    return np.mean(gap)


def plot_max_q_heatmaps(Q_mrl, Q_vanilla, env, save_dir="frozen-lake-plots", filename="max_q_comparison.png"):
    nrow = env.unwrapped.nrow
    ncol = env.unwrapped.ncol

    max_q_mrl = np.max(Q_mrl, axis=1).reshape(nrow, ncol)
    max_q_vanilla = np.max(Q_vanilla, axis=1).reshape(nrow, ncol)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axs[0].imshow(max_q_mrl, cmap="viridis", origin="upper")
    axs[0].set_title("MRL Max Q-values")
    axs[0].set_xticks(range(ncol))
    axs[0].set_yticks(range(nrow))
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(max_q_vanilla, cmap="viridis", origin="upper")
    axs[1].set_title("Vanilla Max Q-values")
    axs[1].set_xticks(range(ncol))
    axs[1].set_yticks(range(nrow))
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = Path(save_dir) / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved side-by-side max Q-value heatmaps: {save_path}")


def plot_q_table_heatmaps(Q_mrl, Q_vanilla, env, save_dir="frozen-lake-plots", prefix="qtable"):
    nrow = env.unwrapped.nrow
    ncol = env.unwrapped.ncol
    n_actions = Q_mrl.shape[1]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    action_names = ["←", "↓", "→", "↑"]

    for action in range(n_actions):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        mrl_vals = Q_mrl[:, action].reshape(nrow, ncol)
        vanilla_vals = Q_vanilla[:, action].reshape(nrow, ncol)

        im0 = axs[0].imshow(mrl_vals, cmap="coolwarm", origin="upper")
        axs[0].set_title(f"MRL Q-values for action {action_names[action]}")
        fig.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(vanilla_vals, cmap="coolwarm", origin="upper")
        axs[1].set_title(f"Vanilla Q-values for action {action_names[action]}")
        fig.colorbar(im1, ax=axs[1])

        for ax in axs:
            ax.set_xticks(range(ncol))
            ax.set_yticks(range(nrow))

        plt.tight_layout()
        save_path = save_dir / f"{prefix}_action{action}_{action_names[action]}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved Q-table heatmap with colorbar: {save_path}")


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


def rollout_mrl_map_from_Q(Q, tau, alpha_m, lo, env, steps=20000, epsilon=0.2, normalize=True, brighten=True):
    nrow, ncol = env.unwrapped.nrow, env.unwrapped.ncol
    acc = np.zeros((nrow, ncol), dtype=np.float64)
    cnt = np.zeros((nrow, ncol), dtype=np.int32)

    def log_pi(q):
        v = np.max(q)
        lse = np.log(np.sum(np.exp((q - v)/tau))) + v/tau
        return (q/tau) - lse

    state, _ = env.reset()
    for _ in range(steps):
        lp = log_pi(Q[state])
        probs = np.exp(lp)
        if np.random.rand() < epsilon:
            a = np.random.randint(Q.shape[1])
        else:
            a = int(np.argmax(Q[state]))

        m_add = alpha_m * np.clip(lp[a], lo, 0.0)   # <= 0
        r, c = divmod(state, ncol)
        acc[r, c] += (-m_add if brighten else m_add)
        cnt[r, c] += 1

        state, _, term, trunc, _ = env.step(a)
        if term or trunc:
            state, _ = env.reset()

    heat = np.divide(acc, np.maximum(cnt, 1), out=np.zeros_like(acc), where=cnt>0)
    if normalize:
        mn, mx = heat.min(), heat.max()
        if mx > mn:
            heat = (heat - mn) / (mx - mn)
    return heat



import numpy as np

def expected_mrl_map_from_Q(Q, tau, alpha_m, lo, env, normalize=True, brighten=True):
    """
    Returns an (nrow, ncol) heatmap from a tabular Q.
    Heat per state s is  - alpha_m * E_{pi}[ clip(log pi(a|s), lo, 0) ].
    (Negated so higher = brighter where policy is sharp.)
    """
    nrow, ncol = env.unwrapped.nrow, env.unwrapped.ncol
    heat = np.zeros((nrow, ncol), dtype=np.float32)

    for s in range(Q.shape[0]):
        q = Q[s]
        # log-softmax with temperature tau
        v = np.max(q)
        logsumexp = np.log(np.sum(np.exp((q - v) / tau))) + v / tau
        log_pi = (q / tau) - logsumexp           # shape (A,)
        pi = np.exp(log_pi)                      # softmax probs

        m_exp = alpha_m * np.sum(pi * np.clip(log_pi, lo, 0.0))  # <= 0
        val = -m_exp if brighten else m_exp

        r, c = divmod(s, ncol)
        heat[r, c] = val

    if normalize:
        mn, mx = heat.min(), heat.max()
        if mx > mn:
            heat = (heat - mn) / (mx - mn)
    return heat




def main():
    # seed = 42
    # set_seed(seed)

    wandb.init(project="frozenlake-cfn", name="MRL-true-vs-pseudo-bonues")

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
        alpha=0.5,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.05,
        tau=0.03,
        alpha_m=0.9,
        lo=-1.0,
        noise_std=0.1
    )

    Q_vanilla, true_counts_vanilla = train_q_learning_without_mrl(
        env=env,
        max_timesteps=50000,
        alpha=0.9,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.05,
        noise_std=0.1
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

    noise = np.random.normal(0, 0.05, size=Q_mrl.shape)
    avg_reward = evaluate_agent(Q_mrl+noise, env, is_save_gif=True)
    print(f"\nAverage evaluation reward over 100 episodes (MRL): {avg_reward:.2f}")

    noise = np.random.normal(0, 0.05, size=Q_mrl.shape)
    avg_reward = evaluate_agent(Q_vanilla+noise, env, is_save_gif=True, gif_path="vanilla.gif")
    print(f"\nAverage evaluation reward over 100 episodes (Vanilla): {avg_reward:.2f}")

    plot_true_vs_munchausen_bonus_heatmap(
        Q=Q_mrl,
        true_counts=true_counts,
        tau=0.03,
        alpha_m=0.9,
        lo=-1.0,
        env=env,
        plotname="munchausen_vs_true_bonus"
    )

    # --- Munchausen state maps from tabular Q ---
    m_exp = expected_mrl_map_from_Q(Q_mrl, tau=0.03, alpha_m=0.9, lo=-1.0, env=env)
    m_roll = rollout_mrl_map_from_Q(Q_mrl, tau=0.03, alpha_m=0.9, lo=-1.0, env=env, steps=30000, epsilon=0.2)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    im0 = ax[0].imshow(m_exp, origin='upper'); ax[0].set_title("Munchausen (expected)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    im1 = ax[1].imshow(m_roll, origin='upper'); ax[1].set_title("Munchausen (rollout)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    Path("frozen-lake-plots").mkdir(parents=True, exist_ok=True)
    plt.savefig(Path("frozen-lake-plots") / "munchausen_state_maps.png")
    plt.close()


    plot_q_table_heatmaps(Q_mrl, Q_vanilla, env, prefix="qtable_mrl_vs_vanilla")

    plot_max_q_heatmaps(Q_mrl, Q_vanilla, env)

    print(true_counts.reshape(int(np.sqrt(len(true_counts))), int(np.sqrt(len(true_counts)))))
    print(true_counts_vanilla.reshape(int(np.sqrt(len(true_counts_vanilla))), int(np.sqrt(len(true_counts_vanilla)))))

    action_symbols = {
        0: '←',
        1: '↓',
        2: '→',
        3: '↑',
    }

    print_policy(Q_mrl, env, action_symbols=action_symbols)
    print_policy(Q_vanilla, env, action_symbols=action_symbols)


if __name__ == "__main__":
    main()
