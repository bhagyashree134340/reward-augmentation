import logging
from pathlib import Path
import numpy as np

from utils.gif import rendered_rollout, save_rgb_animation
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


def validate(actor, env, episode_idx: int, max_steps=500):
    # Get the current Hydra output directory
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    validate_dir = output_dir / "validate" / f"{episode_idx}"
    validate_dir.mkdir(parents=True, exist_ok=True)

    # Run the rollout
    frames, rollout_data = rendered_rollout(actor, env, return_data=True, max_steps=max_steps)

    # Save .npz file
    npz_path = validate_dir / f"rollout_ep{episode_idx:04d}.npz"
    np.savez_compressed(npz_path, **rollout_data)

    # Save gif
    gif_path = validate_dir / f"rollout_ep{episode_idx:04d}.gif"
    save_rgb_animation(frames, gif_path)

    # Save validation stats plot
    rewards = np.array(rollout_data["rewards"])
    timesteps = np.arange(1, len(rewards) + 1)
    cumulative_rewards = np.cumsum(rewards)

    fig, ax = plt.subplots()
    ax.plot(timesteps, cumulative_rewards)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(f"Rollout Reward - Episode {episode_idx}")
    fig.tight_layout()

    plot_path = validate_dir / f"reward_plot_ep{episode_idx:04d}.png"
    fig.savefig(plot_path)
    plt.close(fig)

    log.info(f"Saved validation plot and rollout for episode {episode_idx}")
