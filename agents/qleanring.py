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


def train_q_learning(env, episodes, max_steps, alpha, gamma, epsilon, epsilon_decay, epsilon_min,
                     cfn, cfn_buffer, cfn_optimizer, coin_flip_dim):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Q = np.zeros((state_size, action_size))
    true_counts = np.zeros(state_size, dtype=np.int32)

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        true_counts[state] += 1
        obs_tensor = one_hot(state, state_size)

        for _ in range(max_steps):
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            intrinsic_reward = compute_intrinsic_reward(
                coin_flip_dim,
                cfn.compute_squared_output_norm(obs_tensor)
            )

            _ = cfn(obs_tensor, update_prior_stats=True)

            coin_flip = get_coin_flips(coin_flip_dim)
            cfn_buffer.add(
                obs=obs_tensor.cpu().numpy().squeeze(),
                coin_flip=coin_flip.detach().cpu().numpy(),
                priority=1.0
            )

            best_next_action = np.argmax(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

            obs_batch_bc, coin_flip_batch_bc, indices = cfn_buffer.sample_with_indices(
                batch_size=256
            )

            update_cfn_network(cfn, cfn_optimizer, obs_batch_bc, coin_flip_batch_bc)

            cfn_buffer.update_priorities(
                indices, obs_batch_bc, cfn, coin_flip_dim
            )

            state = next_state
            total_reward += reward
            true_counts[state] += 1
            obs_tensor = one_hot(state, state_size)
            _ = cfn(obs_tensor, update_prior_stats=True)

            if done:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    return Q, true_counts


def evaluate_agent(Q, env, episodes=100, gif_path="frozenlake_qlearning.gif"):
    total_rewards = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_rewards += reward
    save_gif(Q, env, gif_path)
    return total_rewards / episodes


def save_gif(Q, env, filename="frozenlake_qlearning.gif", max_steps=100):
    frames = []
    state, _ = env.reset()
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


def plot_true_vs_pseudo_bonus_normalized(cfn, true_counts, coin_flip_dim, save_path=None):
    state_size = len(true_counts)
    states = list(range(state_size))

    true_bonus = np.array([1 / (c + 1e-8) for c in true_counts])
    approx_bonus = np.array([
        np.sqrt(cfn.compute_squared_output_norm(one_hot(s, state_size)).item() / coin_flip_dim)
        for s in states
    ])

    # Normalize to [0, 1]
    true_bonus = (true_bonus - true_bonus.min()) / (true_bonus.max() - true_bonus.min())
    approx_bonus = (approx_bonus - approx_bonus.min()) / (approx_bonus.max() - approx_bonus.min())

    r = np.corrcoef(true_bonus, approx_bonus)[0, 1]

    plt.figure(figsize=(5, 5))
    plt.scatter(true_bonus, approx_bonus, alpha=0.6, s=20)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("True Bonus")
    plt.ylabel("Approx Bonus")
    plt.title(f"CFN\nPearson r = {r:.3f}")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

    return r


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


def main():
    seed = 42
    set_seed(seed)

    wandb.init(project="frozenlake-cfn", name="true-vs-pseudo-counts")

    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
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

    Q, true_counts = train_q_learning(
        env=env,
        episodes=10000,
        max_steps=100,
        alpha=0.8,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        cfn=cfn,
        cfn_buffer=cfn_buffer,
        cfn_optimizer=cfn_optimizer,
        coin_flip_dim=coin_flip_dim
    )

    avg_reward = evaluate_agent(Q, env)
    print(f"\nAverage evaluation reward over 100 episodes: {avg_reward:.2f}")

    plot_true_vs_pseudo_bonus_normalized(cfn, true_counts, coin_flip_dim)


if __name__ == "__main__":
    main()
