import numpy as np
from PIL import Image
import torch
import logging

log = logging.getLogger(__name__)


def save_rgb_animation(rgb_arrays, filename, duration=50):
    frames = [Image.fromarray((img).astype(np.uint8)) for img in rgb_arrays]
    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    log.info(f"Saved animation to {filename}")


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
