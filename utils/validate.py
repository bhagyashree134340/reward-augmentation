from pathlib import Path
import numpy as np

from utils.actor_io import load_actor
from utils.gif import rendered_rollout, save_rgb_animation
from hydra.core.hydra_config import HydraConfig

from utils.plots import plot_rollout_rewards


def validate_from_checkpoint(actor_class, env, model_path, episode_idx: int, max_steps: int):
    actor = actor_class(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        env.action_space.low,
        env.action_space.high,
    )

    load_actor(actor, model_path)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    validate_dir = output_dir / "validate" / f"{episode_idx:04d}"
    validate_dir.mkdir(parents=True, exist_ok=True)

    frames, rollout_data = rendered_rollout(actor, env, return_data=True, max_steps=max_steps)

    # Save .npz and .gif
    np.savez_compressed(validate_dir / f"rollout_ep{episode_idx:04d}.npz", **rollout_data)
    save_rgb_animation(frames, validate_dir / f"rollout_ep{episode_idx:04d}.gif")
    plot_rollout_rewards(validate_dir / f"rollout_ep{episode_idx:04d}.png", rollout_data["rewards"])
