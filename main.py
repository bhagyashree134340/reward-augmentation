import time
import logging
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig

from agents.sac_agent import SACAgent
from agents.sac_cfn_agent import SACCFNAgent
from agents.td3 import TD3Agent
from agents.td3_cfn import TD3CFNAgent
from utils.env_wrapper import make_env
from utils.misc import set_seed

log = logging.getLogger(__name__)


def create_agent(cfg: DictConfig, env):
    agent_type = cfg.agent.id.lower()

    agent_classes = {
        "sac": SACAgent,
        "sac_cfn": SACCFNAgent,
        "td3": TD3Agent,
        "td3_cfn": TD3CFNAgent,
    }

    if agent_type not in agent_classes:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    agent_cls = agent_classes[agent_type]
    agent_args = {
        "env": env,
        "learning_rate": cfg.agent.lr,
        "gamma": cfg.agent.discount_factor,
        "tau": cfg.agent.tau,
        "batch_size": cfg.agent.batch_size,
    }

    if "sac" in agent_type:
        agent_args.update({
            "maxlen": cfg.agent.replay_buffer_size,
            "target_entropy": cfg.agent.target_entropy,
        })

    if "cfn" in agent_type:
        agent_args["cfn_cfg"] = cfg.cfn

    if agent_type.startswith("td3"):
        agent_args.update({
            "exploration_noise": cfg.agent.exploration_noise,
            "learning_starts": cfg.agent.learning_starts,
            "policy_noise": cfg.agent.policy_noise,
            "noise_clip": cfg.agent.noise_clip,
            "policy_delay": cfg.agent.policy_delay,
        })

    return agent_cls(**agent_args)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set random seed
    # set_seed(cfg.seed)

    # Create output dirs
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    wandb.init(
        project="sac-reward-aug",
        name=f"{cfg.agent.id}_{cfg.env.id}_cfn_{cfg.agent.cfn}",
        config=dict(cfg),
        reinit=True
    )

    # Environment creation
    env = make_env(cfg.env.id,
                   render_mode=cfg.env.render_mode,
                   max_episode_steps=cfg.env.max_episode_steps
                   )
    eval_env = make_env(
                        cfg.env.id,
                        render_mode=cfg.env.render_mode,
                        max_episode_steps=cfg.env.max_episode_steps
                        )

    # Agent creation
    if cfg.agent.cfn:
        agent = SACCFNAgent(
            env,
            eval_env,
            gamma=cfg.agent.discount_factor,
            lr=cfg.agent.lr,
            batch_size=cfg.agent.batch_size,
            tau=cfg.agent.tau,
            maxlen=cfg.agent.replay_buffer_size,
            target_entropy=cfg.agent.target_entropy,
            cfn_cfg=cfg.cfn if cfg.cfn.get("enabled") else None
        )
    else:
        agent = SACAgent(
            env,
            eval_env,
            gamma=cfg.agent.discount_factor,
            lr=cfg.agent.lr,
            batch_size=cfg.agent.batch_size,
            tau=cfg.agent.tau,
            maxlen=cfg.agent.replay_buffer_size,
            target_entropy=cfg.agent.target_entropy,
        )

    # Start training
    # log.info(f"Starting training with {cfg.agent.id.upper()} on {cfg.env.id}")
    start_time = time.time()
    agent.train(
        total_timesteps=cfg.agent.num_env_steps,
        max_steps=cfg.env.max_episode_steps
    )
    elapsed = time.time() - start_time
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    log.info(f"[✓] Training finished in {int(h)}h {int(m)}m {int(s)}s")


if __name__ == "__main__":
    main()