import torch
import gymnasium as gym
from pathlib import Path
import hydra
from omegaconf import DictConfig

from utils.evaluate import evaluate
from utils.misc import set_seed
from agents.sac_agent import SACAgent
from utils.plots import plot_training_stats
import logging

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    op_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    plots_dir = op_dir / "plots"

    # Ensuring the directories exist
    plots_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(cfg.env.id, continuous=cfg.env.continuous, gravity=cfg.env.gravity, render_mode=cfg.env.render_mode)

    log.info(f"Training on {env.spec.id}")
    log.info(f"gamma={cfg.agent.discount_factor} | lr={cfg.agent.lr} | batch_size={cfg.agent.batch_size}")

    agent = SACAgent(
        env,
        gamma=cfg.agent.discount_factor,
        lr=cfg.agent.lr,
        batch_size=cfg.agent.batch_size,
        tau=cfg.agent.tau,
        maxlen=cfg.agent.replay_buffer_size,
        target_entropy=cfg.agent.target_entropy,
        cfn=cfg.cfn if cfg.get("cfn") else None
    )

    stats = agent.train(cfg.agent.num_episodes, cfg.agent.max_steps)

    plot_training_stats(plots_dir / "training_stats.png", stats)
    evaluate()


if __name__ == "__main__":
    main()

