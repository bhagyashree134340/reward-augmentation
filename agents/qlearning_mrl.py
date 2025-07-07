import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import imageio
import gymnasium as gym
from matplotlib import pyplot as plt
import wandb


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


def plot_munchausen_heatmap(Q, env, tau, alpha_m, lo, plotname="munchausen_heatmap"):
    nrow = env.unwrapped.nrow
    ncol = env.unwrapped.ncol
    num_states = Q.shape[0]

    munchausen_rewards = np.zeros(num_states)

    for s in range(num_states):
        q_vals = Q[s]
        q_tensor = torch.tensor(q_vals, dtype=torch.float32)
        probs = F.softmax(q_tensor / tau, dim=0).numpy()
        a = np.argmax(q_vals)
        log_pi = np.log(np.clip(probs[a], 1e-8, 1.0))
        log_pi = np.clip(log_pi, lo, 0.0)
        munchausen_rewards[s] = alpha_m * tau * log_pi

    heatmap = munchausen_rewards.reshape(nrow, ncol)

    plt.figure(figsize=(8, 6))
    plt.title("Munchausen Shaping Rewards (Per State)")
    plt.imshow(heatmap, cmap="viridis", origin="upper")
    plt.colorbar(label="Munchausen Reward")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.tight_layout()

    save_path = Path("frozen-lake-plots") / f"{plotname}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"Munchausen heatmap saved to: {save_path}")
    plt.close()


def train_q_learning(env, max_timesteps, alpha, tau, gamma, epsilon, epsilon_decay, epsilon_min, alpha_m, lo):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Q = np.zeros((state_size, action_size))
    true_counts = np.zeros(state_size, dtype=np.int32)

    def stable_log_pi(q_values, tau):
        v = np.max(q_values)
        logsumexp = np.log(np.sum(np.exp((q_values - v) / tau)))
        log_pi = (q_values - v) / tau - logsumexp
        return tau * log_pi

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

        log_pi_all = stable_log_pi(Q[state], tau)
        log_pi_a = np.clip(log_pi_all[action], lo, 0.0)

        next_log_pi_all = stable_log_pi(Q[next_state], tau)
        soft_v_next = np.sum(
            np.exp(next_log_pi_all / tau) * (Q[next_state] - next_log_pi_all)
        )

        target = reward + tau * log_pi_a + gamma * soft_v_next

        Q[state][action] += alpha * (target - Q[state][action])

        total_timesteps += 1
        episode_reward += reward
        episode_length += 1
        state = next_state
        true_counts[state] += 1

        if done:
            print(
                f"Timestep {total_timesteps}, "
                f"Return {episode_reward:.2f}, Length {episode_length}"
            )
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            state, _ = env.reset()
            true_counts[state] += 1
            episode_reward = 0
            episode_length = 0

    return Q, true_counts


def softmax_probs(q_vals, entropy_tau):
    q_tensor = torch.tensor(q_vals, dtype=torch.float32)
    probs_tensor = F.softmax(q_tensor / entropy_tau, dim=0)
    return probs_tensor.numpy()


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


def main():
    # seed = 42
    # set_seed(seed)

    wandb.init(project="frozenlake-cfn", name="MRL-true-vs-pseudo-bonues")

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
    state_size = env.observation_space.n

    Q, true_counts = train_q_learning(
        env=env,
        max_timesteps=100000,
        alpha=1.0,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.05,
        tau=0.03,
        alpha_m=1.0,
        lo=-1.0
    )

    print("Q-values at state 0:")
    for a in range(env.action_space.n):
        print(f"Action {a}: {Q[0, a]:.4f}")

    avg_reward = evaluate_agent(Q, env)
    print(f"\nAverage evaluation reward over 100 episodes: {avg_reward:.2f}")

    plot_munchausen_heatmap(Q, env, tau=0.05, alpha_m=0.3, lo=-0.1, plotname="munchausen_heatmap_run1")

    print(true_counts.reshape(int(np.sqrt(len(true_counts))), int(np.sqrt(len(true_counts)))))

    action_symbols = {
        0: '←',
        1: '↓',
        2: '→',
        3: '↑',
    }

    print_policy(Q, env, tau=0.03, action_symbols=action_symbols)

if __name__ == "__main__":
    main()
