import torch
import gymnasium as gym
from pathlib import Path
from IPython.display import Image as IImage
import hydra
from omegaconf import DictConfig

from utils.misc import set_seed
from networks.network import Actor
from agents.sac_agent import SACAgent
from utils.gif import rendered_rollout, save_rgb_animation
from utils.plots import plot_training_stats
import logging

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    op_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    plots_dir = op_dir / "plots"
    gifs_dir = op_dir / "gifs"

    # Ensuring the directories exist
    plots_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(cfg.env.id, continuous=cfg.env.continuous, gravity=cfg.env.gravity, render_mode=cfg.env.render_mode)
    log.info(f"Training on {env.spec.id}")
    # log.info(f"Observation space: {env.observation_space}")
    # log.info(f"Action space: {env.action_space}\n")
    log.info(f"gamma={cfg.agent.discount_factor} | lr={cfg.agent.lr} | batch_size={cfg.agent.batch_size}")

    agent = SACAgent(
        env,
        gamma=cfg.agent.discount_factor,
        lr=cfg.agent.lr,
        batch_size=cfg.agent.batch_size,
        tau=cfg.agent.tau,
        maxlen=cfg.agent.replay_buffer_size,
        target_entropy=cfg.agent.target_entropy
    )

    stats = agent.train(cfg.agent.num_episodes, cfg.agent.max_steps)

    # Save and load actor
    actor_path = op_dir / "sac_actor.pt"
    torch.save(agent.actor, actor_path)

    # Add Actor to safe globals for torch.load()
    torch.serialization.add_safe_globals([Actor])

    loaded_actor = torch.load(actor_path, weights_only=False)
    loaded_actor.eval()
    log.info(f"Saved and loaded actor from {actor_path}")

    # Plot stats and save gif
    plot_training_stats(plots_dir / "training_stats.png", stats)
    imgs = rendered_rollout(loaded_actor, env)
    gif_path = gifs_dir / "trained.gif"
    save_rgb_animation(imgs, gif_path)
    IImage(filename=str(gif_path))


if __name__ == "__main__":
    main()
