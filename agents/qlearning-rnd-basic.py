from collections import deque
from pathlib import Path

import numpy as np
import torch
import imageio
import gymnasium as gym
from gymnasium.wrappers.utils import RunningMeanStd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import wandb
import random

from RND.rnd import RNDModel, RewardForwardFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "wandb_project": "frozenlake-cfn",
    "wandb_run_name": "true-vs-pseudo-counts",

    "env": {
        "is_slippery": False,
        "render_mode": "rgb_array",
        "max_episode_steps": 1000,
        "map": [
            "SFFFFFFH",
            "HHHHFFFH",
            "FFFFFHFF",
            "FGFFFFFH",
            "FHFFFHFF",
            "FHFFFFHF",
            "FFFFHHHF",
            "HHHHHFFG",
        ],
    },

    "rnd": {
        "lr": 1e-6,
        "intrinsic_coef": 1.0,
        "rnd_mask_prob": 0.75,
    },

    "q_learning": {
        "max_timesteps": 200_000,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.05,
    },

    "vanilla_q": {
        "max_timesteps": 200_000,
        "alpha": 0.1,
        "gamma": 0.999,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.05,
        "buffer_size": 200_000,
        "batch_size": 64,
    }
}


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def one_hot(state, size):
    vec = torch.zeros(size, device=device)
    vec[state] = 1.0

    return vec.unsqueeze(0)


def train_q_learning(env, rnd,
                     rnd_optimizer,
                     obs_rms,
                     reward_rms,
                     discounted_reward,
                     max_timesteps, alpha, gamma, epsilon, epsilon_decay, epsilon_min,
                     intrinsic_coef, rnd_mask_prob=0.75):  # 0.25
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Q = np.zeros((state_size, action_size))
    true_counts = np.zeros(state_size, dtype=np.int32)

    obs_buffer = deque(maxlen=100_000)
    BATCH_SIZE = 128

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
        next_state_tensor = one_hot(next_state, state_size).float().to(device)
        obs_rms.update(next_state_tensor.unsqueeze(0).cpu().numpy())

        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    state, _ = env.reset()
    state_tensor = one_hot(state, state_size).float().to(device)
    obs_rms.update(state_tensor.numpy())
    true_counts[state] += 1

    while total_timesteps < max_timesteps:
        if np.random.rand() < epsilon or np.all(Q[state] == 0):
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, ext_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done and next_state == 63:
            ext_reward = 10.0

        next_state_tensor = one_hot(next_state, state_size).float()
        obs_rms.update(next_state_tensor.unsqueeze(0).cpu().numpy())
        obs_buffer.append(next_state_tensor.detach())

        mean_tensor = torch.from_numpy(obs_rms.mean).float().to(device)
        var_tensor = torch.from_numpy(obs_rms.var).float().to(device)
        normalized_next_obs = ((next_state_tensor - mean_tensor) / torch.sqrt(var_tensor + 1e-8))

        with torch.no_grad():
            target = rnd.target(normalized_next_obs)
        pred = rnd.predictor(normalized_next_obs)

        int_reward = 0.5 * ((pred - target) ** 2).sum().detach()
        sample_i_rall += int_reward

        current_discounted_intrinsic_return = discounted_reward.update(int_reward.item())

        reward_rms.update(np.array([[current_discounted_intrinsic_return]]))
        normalized_int_reward = float(
            int_reward / np.sqrt(float(np.maximum(reward_rms.var.item(), 1e-8)))
        )

        wandb.log({
            "int_rew_norm": intrinsic_coef * normalized_int_reward,
            "ext_rew": ext_reward
        }, step=total_timesteps)
        log_statewise_data(next_state, normalized_int_reward, total_timesteps, "int_reward")

        total_reward = ext_reward + intrinsic_coef * normalized_int_reward

        best_next_action = np.argmax(Q[next_state])
        Q[state, action] += alpha * (
                float(total_reward) + gamma * Q[next_state, best_next_action] - Q[state, action]
        )

        if len(obs_buffer) >= BATCH_SIZE:
            batch = torch.stack(random.sample(obs_buffer, BATCH_SIZE)).to(device)
            norm_batch = torch.cat([next_state_tensor.unsqueeze(0), batch], dim=0)
            mean_tensor = torch.from_numpy(obs_rms.mean).float().to(batch.device)
            var_tensor = torch.from_numpy(obs_rms.var).float().to(batch.device)
            norm_batch = (norm_batch - mean_tensor) / torch.sqrt(var_tensor + 1e-8)

            with torch.no_grad():
                target_batch = rnd.target(norm_batch)
            pred_batch = rnd.predictor(norm_batch)

            fwd_loss = torch.nn.functional.mse_loss(pred_batch, target_batch, reduction='none').mean(dim=-1)
            mask = (torch.rand_like(fwd_loss) < rnd_mask_prob).float()
            loss = (fwd_loss * mask).sum() / mask.sum().clamp(min=1.0)

            rnd_optimizer.zero_grad()
            loss.backward()
            rnd_optimizer.step()

        if total_timesteps % 5000 == 0 and total_timesteps > 0:
            avg_reward = evaluate_agent(Q, env, episodes=20)
            wandb.log({"eval/avg_reward": avg_reward}, step=total_timesteps)

        if total_timesteps % 10000 == 0 and total_timesteps > 0:
            plot_intrinsic_vs_true_bonus_heatmap(
                rnd, intrinsic_coef, true_counts, reward_rms, obs_rms,
                step=total_timesteps, save_dir="rnd"
            )

        total_timesteps += 1
        episode_reward += ext_reward
        episode_length += 1

        state = next_state
        true_counts[state] += 1

        if done:
            sample_episode += 1
            print(
                f"Timestep {total_timesteps}, Episode {sample_episode}, "
                f"Epsilon {epsilon:.3f}, Return {ext_reward:.2f}, "
                f"IntReward {intrinsic_coef * normalized_int_reward}, Length {episode_length}"
            )

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            state, _ = env.reset()
            true_counts[state] += 1
            state_tensor = one_hot(state, state_size).float().to(device)
            obs_rms.update(state_tensor.unsqueeze(0).cpu().numpy())

            episode_reward = 0
            episode_length = 0
            sample_i_rall = 0

    return Q, true_counts


def log_statewise_data(state, data, step, prefix):
    wandb.log({f"{prefix}/state_{state}": data}, step=step)


def evaluate_agent(Q, env, episodes=100, save_gif_at_end=False, gif_path="rnd/policy_final.gif"):
    total_rewards = 0

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_rewards += episode_reward

    if save_gif_at_end:
        save_gif(Q, env, filename=Path(gif_path).name, save_dir=str(Path(gif_path).parent))

    return total_rewards / episodes


def save_gif(Q, env, filename="frozenlake_qlearning.gif", max_steps=700, save_dir="rnd"):
    frames = []
    state, _ = env.reset()

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    filename = save_path / f"{filename}"

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


def plot_true_bonus_heatmap(true_counts, step, save_dir="rnd/plots/vanilla"):
    state_size = len(true_counts)
    grid_size = int(np.sqrt(state_size))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"true_bonus_timestep_{step:06d}.png"

    true_bonus = 1.0 / np.sqrt(true_counts + 1e-8)

    visited_mask = true_counts > 0
    outlier_mask = true_bonus < 1e3
    valid_mask = visited_mask & outlier_mask

    masked_true_bonus = np.ma.masked_where(~valid_mask, true_bonus)
    masked_true_bonus_grid = masked_true_bonus.reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(masked_true_bonus_grid, cmap="magma")
    fig.colorbar(im, ax=ax)
    ax.set_title("True Bonus")
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xticklabels(range(grid_size))
    ax.set_yticklabels(range(grid_size))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Step {step}] Saved true bonus heatmap to {save_path}")


def plot_intrinsic_vs_true_bonus_heatmap(rnd, intrinsic_coef, true_counts, reward_rms, obs_rms,
                                         step, save_dir="rnd"):
    state_size = len(true_counts)
    grid_size = int(np.sqrt(state_size))

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"timestep_{step:06d}.png"

    mean_tensor = torch.from_numpy(obs_rms.mean).float().to(device)
    var_tensor = torch.from_numpy(obs_rms.var).float().to(device)

    rnd_bonus = []
    for s in range(state_size):
        obs = one_hot(s, state_size).to(device)
        normalized_obs = (obs - mean_tensor) / torch.sqrt(var_tensor + 1e-8)
        with torch.no_grad():
            target = rnd.target(normalized_obs)
            pred = rnd.predictor(normalized_obs)
        bonus = ((pred - target) ** 2).sum().item()
        rnd_bonus.append(bonus)

    rnd_bonus = np.array(rnd_bonus)
    norm_rnd_bonus = rnd_bonus / np.sqrt(np.maximum(reward_rms.var, 1e-8))
    scaled_rnd_bonus = intrinsic_coef * norm_rnd_bonus

    true_bonus = 1.0 / np.sqrt(true_counts + 1e-8)
    visited_mask = (true_counts > 0)

    masked_true_bonus = np.ma.masked_where(~visited_mask, true_bonus)
    masked_approx_bonus = np.ma.masked_where(~visited_mask, scaled_rnd_bonus)

    true_bonus_grid = masked_true_bonus.reshape(grid_size, grid_size)
    approx_bonus_grid = masked_approx_bonus.reshape(grid_size, grid_size)

    scatter_true = masked_true_bonus.compressed()
    scatter_approx = masked_approx_bonus.compressed()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    cmaps = ["magma", "magma"]
    titles = ["True Bonus", "Approx Bonus"]

    for ax, data, title, cmap in zip(axs[:2], [true_bonus_grid, approx_bonus_grid], titles, cmaps):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_xticklabels(range(grid_size))
        ax.set_yticklabels(range(grid_size))

    axs[2].scatter(scatter_true, scatter_approx)
    axs[2].set_xlabel("True Bonus")
    axs[2].set_ylabel("Approx Bonus")
    axs[2].set_title("True vs. Approx Bonus")
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Step {step}] Saved plot to {save_path}")


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

        if total_timesteps % 5000 == 0 and total_timesteps > 0:
            avg_reward = evaluate_agent(Q, env, episodes=20)
            wandb.log({"eval/avg_reward_van": avg_reward, "eval/step_van": total_timesteps})

        if total_timesteps % 10000 == 0 and total_timesteps > 0:
            plot_true_bonus_heatmap(true_counts, total_timesteps)

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


def main():
    set_seed(42)

    wandb.init(project=CONFIG["wandb_project"], name=CONFIG["wandb_run_name"])

    env = gym.make(
        "FrozenLake-v1",
        is_slippery=CONFIG["env"]["is_slippery"],
        render_mode=CONFIG["env"]["render_mode"],
        desc=CONFIG["env"]["map"],
        max_episode_steps=CONFIG["env"]["max_episode_steps"]
    )

    state_size = env.observation_space.n

    rnd = RNDModel(state_size).to(device=device)
    rnd_optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=CONFIG["rnd"]["lr"])
    obs_rms = RunningMeanStd(shape=env.observation_space.shape)
    reward_rms = RunningMeanStd()
    discounted_reward = RewardForwardFilter(0.99)

    Q, true_counts = train_q_learning(
        env,
        rnd,
        rnd_optimizer,
        obs_rms,
        reward_rms,
        discounted_reward,
        intrinsic_coef=CONFIG["rnd"]["intrinsic_coef"],
        max_timesteps=CONFIG["q_learning"]["max_timesteps"],
        alpha=CONFIG["q_learning"]["alpha"],
        gamma=CONFIG["q_learning"]["gamma"],
        epsilon=CONFIG["q_learning"]["epsilon"],
        epsilon_decay=CONFIG["q_learning"]["epsilon_decay"],
        epsilon_min=CONFIG["q_learning"]["epsilon_min"],
        rnd_mask_prob=CONFIG["rnd"]["rnd_mask_prob"]
    )

    plot_intrinsic_vs_true_bonus_heatmap(
        rnd, CONFIG["rnd"]["intrinsic_coef"], true_counts, reward_rms, obs_rms,
        step=CONFIG["q_learning"]["max_timesteps"], save_dir="rnd"
    )

    final_avg_reward = evaluate_agent(Q, env, episodes=100, save_gif_at_end=True, gif_path="rnd/policy_final.gif")
    print(f"Final average reward: {final_avg_reward}")

    print("vanilla:")

    Q_vanilla, true_counts_vanilla = train_q_learning_vanilla(
        env=env,
        max_timesteps=CONFIG["vanilla_q"]["max_timesteps"],
        alpha=CONFIG["vanilla_q"]["alpha"],
        gamma=CONFIG["vanilla_q"]["gamma"],
        epsilon=CONFIG["vanilla_q"]["epsilon"],
        epsilon_decay=CONFIG["vanilla_q"]["epsilon_decay"],
        epsilon_min=CONFIG["vanilla_q"]["epsilon_min"],
        buffer_size=CONFIG["vanilla_q"]["buffer_size"],
        batch_size=CONFIG["vanilla_q"]["batch_size"]
    )

    plot_true_bonus_heatmap(true_counts_vanilla, CONFIG["vanilla_q"]["max_timesteps"])

    print(true_counts.reshape(int(np.sqrt(len(true_counts))), -1))
    print(true_counts_vanilla.reshape(int(np.sqrt(len(true_counts_vanilla))), -1))


if __name__ == "__main__":
    main()
