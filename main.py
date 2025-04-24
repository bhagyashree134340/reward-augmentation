import torch
import numpy as np
import pandas as pd
from PIL import Image
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Image as IImage

from network import Actor
from sac_agent import SACAgent

# Directory to save outputs
OUTPUT_DIR = Path("outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
GIFS_DIR = OUTPUT_DIR / "gifs"
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
GIFS_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_stats(stats, smoothing_window=20):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

    axes[0].plot(stats.episode_lengths)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode Length")
    axes[0].set_title("Episode Length over Time")

    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    axes[1].plot(rewards_smoothed)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode Reward (Smoothed)")
    axes[1].set_title(f"Episode Reward over Time\n(Smoothed over window size {smoothing_window})")

    plot_path = PLOTS_DIR / "training_stats.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Saved training plots to {plot_path}")


def save_rgb_animation(rgb_arrays, filename, duration=50):
    frames = [Image.fromarray((img).astype(np.uint8)) for img in rgb_arrays]
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    print(f"Saved animation to {filename}")


def rendered_rollout(policy, env, max_steps=1_000):
    obs, _ = env.reset()
    imgs = [env.render()]
    for _ in range(max_steps):
        with torch.no_grad():
            action = policy(torch.as_tensor(obs, dtype=torch.float32))[0].cpu().numpy()
        obs, _, terminated, truncated, _ = env.step(action)
        imgs.append(env.render())
        if terminated or truncated:
            break
    return imgs


def main():
    env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0, render_mode="rgb_array")
    print(f"Training on {env.spec.id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}\n")

    # Hyperparameters
    # TODO: Add a config for this
    LR = 0.001
    BATCH_SIZE = 32
    REPLAY_BUFFER_SIZE = 100_000
    TAU = 0.005
    NUM_EPISODES = 1_000
    DISCOUNT_FACTOR = 0.99
    TARGET_ENTROPY = -1.0

    agent = SACAgent(env, gamma=DISCOUNT_FACTOR, lr=LR, batch_size=BATCH_SIZE,
                     tau=TAU, maxlen=REPLAY_BUFFER_SIZE, target_entropy=TARGET_ENTROPY)
    stats = agent.train(NUM_EPISODES)

    # Save and load actor
    actor_path = OUTPUT_DIR / "sac_actor.pt"
    torch.save(agent.actor, actor_path)

    # Add Actor to safe globals for torch.load()
    torch.serialization.add_safe_globals([Actor])

    loaded_actor = torch.load(actor_path, weights_only=False)
    loaded_actor.eval()
    print(f"Saved and loaded actor from {actor_path}")

    # Plot stats and save gif
    plot_training_stats(stats)
    imgs = rendered_rollout(loaded_actor, env)
    gif_path = GIFS_DIR / "trained.gif"
    save_rgb_animation(imgs, gif_path)
    IImage(filename=str(gif_path))


if __name__ == "__main__":
    main()
