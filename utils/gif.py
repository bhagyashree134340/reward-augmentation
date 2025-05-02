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

