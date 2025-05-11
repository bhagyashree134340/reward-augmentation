import numpy as np
from PIL import Image
import torch
import logging

log = logging.getLogger(__name__)


def save_rgb_animation(rgb_arrays, filename, duration=50):
    frames = [Image.fromarray((img).astype(np.uint8)) for img in rgb_arrays]
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    log.info(f"Saved animation to {filename}")


def rendered_rollout(actor, env, return_data=False, max_steps=1000):
    obs, _ = env.reset()
    done = False
    frames = []
    data = {"observations": [], "actions": [], "rewards": []}

    for _ in range(max_steps):
        frames.append(env.render())

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, _ = actor(obs_tensor)
            action = action.cpu().numpy()[0]

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if return_data:
            data["observations"].append(obs)
            data["actions"].append(action)
            data["rewards"].append(reward)

        if done:
            break
        obs = next_obs

    return (frames, data) if return_data else frames


def evaluate_policy(actor, env, num_episodes: int = 10, max_steps: int = 1000):
    """
    Evaluates the policy over a number of episodes and stores data similar to SB3's EvalCallback.
    """
    actor.eval()

    all_obs = []
    all_actions = []
    all_rewards = []
    episode_lengths = []
    episode_rewards = []
    timesteps = []

    total_timesteps = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        length = 0

        while not done and length < max_steps:
            all_obs.append(obs)

            with torch.no_grad():
                action, _ = actor(torch.as_tensor(obs).float())
                action = action.cpu().numpy().clip(env.action_space.low, env.action_space.high)
            all_actions.append(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            all_rewards.append(reward)

            total_reward += reward
            done = terminated or truncated
            length += 1
            total_timesteps += 1

        episode_lengths.append(length)
        episode_rewards.append(total_reward)
        timesteps.append(total_timesteps)

    data = {
        "episode_lengths": np.array(episode_lengths),
        "episode_rewards": np.array(episode_rewards),
        "timesteps": np.array(timesteps),
        "observations": np.array(all_obs),
        "actions": np.array(all_actions),
        "rewards": np.array(all_rewards),
    }

    return data


def save_rollout_gif(actor, env, gif_path, max_steps=1000, duration=50):
    frames = rendered_rollout(actor, env, return_data=False, max_steps=max_steps)
    save_rgb_animation(frames, gif_path, duration=duration)