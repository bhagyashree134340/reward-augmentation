import imageio
import numpy as np
from PIL import Image
import torch
import logging

log = logging.getLogger(__name__)


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


def save_rollout_gif(actor, env, gif_path, max_episode_steps=1000):
    import imageio
    import torch
    import numpy as np

    frames = []
    obs, _ = env.reset()
    done = False
    step = 0
    actor.eval()

    env.render()

    if "Fetch" in env.spec.id:
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        try:
            cam = base_env.mujoco_renderer.viewer.cam
            cam.distance = 5
            cam.lookat[:] = [1.3, 0.75, 0.5]
            cam.azimuth = 180
            cam.elevation = -20
        except Exception as e:
            log.info("Failed to adjust Fetch camera settings:", e)

    while not done and step < max_episode_steps:
        frame = env.render()
        frames.append(frame)

        obs_tensor = torch.FloatTensor(
            obs if not isinstance(obs, dict)
            else np.concatenate([v.flatten() for v in obs.values()])
        ).unsqueeze(0)

        with torch.no_grad():
            action, _ = actor(obs_tensor)
            action = action.cpu().numpy()[0]

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1

    actor.train()
    imageio.mimsave(gif_path, frames, fps=30)
