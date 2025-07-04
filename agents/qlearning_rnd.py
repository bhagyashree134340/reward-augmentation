from pathlib import Path

import numpy as np
import torch
import imageio
import gymnasium as gym
from gymnasium.wrappers.utils import RunningMeanStd
from matplotlib import pyplot as plt
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
                     discounted_reward,
                     max_timesteps, alpha, gamma, epsilon, epsilon_decay, epsilon_min,
                     intrinsic_coef, rnd_mask_prob=0.5):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Q = np.zeros((state_size, action_size))
    true_counts = np.zeros(state_size, dtype=np.int32)

    total_timesteps = 0
    episode_reward = 0
    episode_length = 0
    sample_i_rall = 0
    sample_episode = 0
    normed_int_reward = 0

    state, _ = env.reset()

    for _ in range(5000):
        random_action = env.action_space.sample()
        next_state, _, terminated, truncated, _ = env.step(random_action)
        next_state_tensor = one_hot(next_state, state_size)
        obs_rms.update(next_state_tensor.unsqueeze(0).cpu().numpy())

        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    state, _ = env.reset()
    true_counts[state] += 1

    while total_timesteps < max_timesteps:
        if np.random.rand() < epsilon or np.all(Q[state] == 0):
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, ext_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state_tensor = one_hot(next_state, state_size).float()

        obs_rms.update(next_state_tensor.unsqueeze(0).numpy())

        mean_tensor = torch.from_numpy(obs_rms.mean).float()
        var_tensor = torch.from_numpy(obs_rms.var).float()

        normalized_next_obs = ((next_state_tensor - mean_tensor) / torch.sqrt(var_tensor + 1e-8))

        with torch.no_grad():
            target = rnd.target(normalized_next_obs)
        pred = rnd.predictor(normalized_next_obs)

        int_reward = ((pred - target) ** 2).sum().detach()

        # int_reward_np = int_reward.detach().cpu().numpy()

        sample_i_rall += int_reward

        reward_rms.update(np.array([[int_reward]]))

        # TODO: normed_int_reward is reaching values like 200

        # After computing int_reward = ((pred - target) ** 2).sum().detach()
        normalized_int_reward = float(
            int_reward / np.sqrt(np.maximum(reward_rms.var, 1e-8))
        )
        normalized_int_reward = max(0.0, min(1.0, normalized_int_reward))

        wandb.log({
            "int_rew_norm": intrinsic_coef * normalized_int_reward,
            "ext_rew": ext_reward
        }, step=total_timesteps)

        total_reward = ext_reward + intrinsic_coef * normalized_int_reward

        best_next_action = np.argmax(Q[next_state])
        Q[state, action] += alpha * (
                float(total_reward) + gamma * Q[next_state, best_next_action] - Q[state, action]
        )

        # with torch.no_grad():
        #     target_next_state_feature = rnd.target(normalized_next_obs)
        # predict_next_state_feature = rnd.predictor(normalized_next_obs)

        # TODO: plot loss over timesteps
        forward_loss = torch.nn.functional.mse_loss(pred, target, reduction='sum')

        wandb.log({
            "rnd loss": forward_loss
        }, step=total_timesteps)

        if np.random.rand() < rnd_mask_prob:
            rnd_optimizer.zero_grad()
            forward_loss.backward()
            rnd_optimizer.step()

        total_timesteps += 1
        episode_reward += ext_reward
        episode_length += 1

        state = next_state
        true_counts[state] += 1

        if done:
            sample_episode += 1
            # normed_int_reward[0] = 0
            print(
                f"Timestep {total_timesteps}, Episode {sample_episode}, "
                f"Epsilon {epsilon:.3f}, Return {ext_reward:.2f}, "
                f"IntReward {intrinsic_coef * normalized_int_reward}, Length {episode_length}"
            )

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            state, _ = env.reset()
            true_counts[state] += 1
            state_tensor = one_hot(state, state_size).float()
            obs_rms.update(state_tensor.unsqueeze(0).cpu().numpy())

            episode_reward = 0
            episode_length = 0
            sample_i_rall = 0

    return Q, true_counts


def evaluate_agent(Q, env, episodes=100, gif_path="frozenlake_qlearning_20x20.gif"):
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


def plot_intrinsic_vs_true_bonus_heatmap(rnd, intrinsic_coef, true_counts, reward_rms, obs_rms,
                                         save_dir="frozen-lake-plots"):
    state_size = len(true_counts)
    grid_size = int(np.sqrt(state_size))
    save_path = Path(save_dir) / f"frozenlake_qlearning_rnd_{grid_size}x{grid_size}_heatmap.png"
    mean_tensor = torch.from_numpy(obs_rms.mean).float()
    var_tensor = torch.from_numpy(obs_rms.var).float()

    rnd_bonus = []
    for s in range(state_size):
        obs = one_hot(s, state_size)
        normalized_obs = (obs - mean_tensor) / torch.sqrt(var_tensor + 1e-8)
        with torch.no_grad():
            target = rnd.target(normalized_obs)
            pred = rnd.predictor(normalized_obs)
        bonus = ((pred - target) ** 2).sum().item()
        rnd_bonus.append(bonus)

    rnd_bonus = np.array(rnd_bonus)
    norm_rnd_bonus = rnd_bonus / np.sqrt(np.maximum(reward_rms.var, 1e-8))
    norm_rnd_bonus = np.clip(norm_rnd_bonus, 0.0, 1.0)
    scaled_rnd_bonus = intrinsic_coef * norm_rnd_bonus

    true_bonus = 1.0 / np.sqrt(true_counts + 1e-8)
    visited_mask = (true_counts > 0)
    masked_true_bonus = true_bonus[visited_mask]
    masked_approx_bonus = scaled_rnd_bonus[visited_mask]

    outlier_mask = masked_true_bonus < 1e3
    masked_true_bonus_grid = np.ma.masked_where(~visited_mask, true_bonus).reshape(grid_size, grid_size)
    norm_rnd_grid = scaled_rnd_bonus.reshape(grid_size, grid_size)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    cmaps = ["magma", "magma"]
    titles = ["True Bonus", "Approx Bonus"]

    for ax, data, title, cmap in zip(axs[:2], [masked_true_bonus_grid, norm_rnd_grid], titles, cmaps):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_xticklabels(range(grid_size))
        ax.set_yticklabels(range(grid_size))

    axs[2].scatter(masked_true_bonus[outlier_mask], masked_approx_bonus[outlier_mask])
    axs[2].set_xlabel("True Bonus")
    axs[2].set_ylabel("Approx Bonus")
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
    map = [
        "SFFFFFFHFFFFFFFF",
        "FHFFFFFHFFFFFHFF",
        "FFFHFFFFFFFHFFFF",
        "FHHFFFHFFFHFFFFF",
        "FFFHFFFFFFFFHFFF",
        "FFFFFHFFFFHFHFFF",
        "FHFHFFFFFFFFFHFF",
        "FFFFFHFFFFFHFHFF",
        "FHFFFFFFHFFFFHFF",
        "FFFFFHFFFHFFFFHF",
        "FHFFFFHFFFHFFFFF",
        "FFFFFHFFFFFFHFFF",
        "FFHFHFFFFHFHFHFF",
        "FFFHFFFFFHFFFFFH",
        "FFFFHFHFFFFFFHFF",
        "FFFFFFFFFFFFFFFG"
    ]

    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array", desc=map, max_episode_steps=1000)
    state_size = env.observation_space.n
    rnd = RNDModel(env.observation_space.n)
    rnd_optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-4)
    obs_rms = RunningMeanStd(shape=env.observation_space.shape)
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
        intrinsic_coef=intrinsic_coef,
        max_timesteps=100000,
        alpha=0.1,
        gamma=0.999,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.05
    )

    print(true_counts.reshape(int(np.sqrt(len(true_counts))), int(np.sqrt(len(true_counts)))))

    plot_intrinsic_vs_true_bonus_heatmap(rnd, intrinsic_coef, true_counts, reward_rms, obs_rms)

    avg_reward = evaluate_agent(Q, env)
    print(f"\nAverage evaluation reward over 100 episodes: {avg_reward:.2f}")


if __name__ == "__main__":
    main()
