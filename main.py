import time
import logging
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig

from agents.sac_agent import SACAgent
from agents.sac_cfn_agent import SACCFNAgent
from agents.sac_mrl_agent import SACMRLAgent
from agents.sac_rnd_agent import SACRNDAgent
from agents.td3 import TD3Agent
from agents.td3_cfn import TD3CFNAgent
from utils.env_wrapper import make_env
from utils.misc import set_seed

log = logging.getLogger(__name__)


def create_agent(cfg: DictConfig, env):
    agent_id = cfg.agent.id.lower()
    use_cfn = cfg.agent.cfn
    eval_env = make_env(cfg.env.id,
                        render_mode=cfg.env.render_mode,
                        max_episode_steps=cfg.env.max_episode_steps)

    if agent_id == "sac_agent":
        if use_cfn:
            agent = SACCFNAgent(
                env=env,
                lr=cfg.agent.lr,
                gamma=cfg.agent.discount_factor,
                tau=cfg.agent.tau,
                batch_size=cfg.agent.batch_size,
                maxlen=cfg.agent.replay_buffer_size,
                target_entropy=cfg.agent.target_entropy,
                cfn_cfg=cfg.cfn,
                eval_env=eval_env
            )
        elif cfg.agent.rnd:
            print("SAC RND agent")
            agent = SACRNDAgent(
                env=env,
                eval_env=eval_env,
                gamma=cfg.agent.discount_factor,
                tau=cfg.agent.tau,
                batch_size=cfg.agent.batch_size,
                maxlen=cfg.agent.replay_buffer_size,
                target_entropy=cfg.agent.target_entropy,
                rnd_cfg=cfg.rnd  
        )
        elif cfg.agent.mrl:
            print("SAC MRL agent")
            agent = SACMRLAgent(
                env=env,
                eval_env=eval_env,
                gamma=cfg.agent.discount_factor,
                tau=cfg.agent.tau,
                batch_size=cfg.agent.batch_size,
                maxlen=cfg.agent.replay_buffer_size,
                target_entropy=cfg.agent.target_entropy,
                alpha=cfg.mrl.alpha,
                tau_m=cfg.mrl.tau,
                lo=cfg.mrl.lo,
            )
        else:
            agent = SACAgent(
                env=env,
                gamma=cfg.agent.discount_factor,
                tau=cfg.agent.tau,
                batch_size=cfg.agent.batch_size,
                maxlen=cfg.agent.replay_buffer_size,
                target_entropy=cfg.agent.target_entropy,
                eval_env=eval_env
            )

    elif agent_id == "td3_agent":
        if use_cfn:
            agent = TD3CFNAgent(
                env=env,
                cfn_cfg=cfg.cfn,
                learning_rate=cfg.agent.lr,
                buffer_size=cfg.agent.buffer_size,
                gamma=cfg.agent.discount_factor,
                tau=cfg.agent.tau,
                batch_size=cfg.agent.batch_size,
                exploration_noise=cfg.agent.exploration_noise,
                learning_starts=cfg.agent.learning_starts,
                policy_frequency=cfg.agent.policy_frequency,
                noise_clip=cfg.agent.noise_clip,
                eval_env=eval_env
            )
        else:
            agent = TD3Agent(
                env=env,
                learning_rate=cfg.agent.lr,
                buffer_size=cfg.agent.buffer_size,
                gamma=cfg.agent.discount_factor,
                tau=cfg.agent.tau,
                batch_size=cfg.agent.batch_size,
                exploration_noise=cfg.agent.exploration_noise,
                learning_starts=cfg.agent.learning_starts,
                policy_frequency=cfg.agent.policy_frequency,
                noise_clip=cfg.agent.noise_clip,
            )

    else:
        raise ValueError(f"Unsupported agent id: {agent_id}")

    return agent


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set random seed
    # set_seed(cfg.seed)

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rew_aug = "cfn" if cfg.agent.cfn else "rnd" if cfg.agent.rnd else "mrl" if cfg.agent.mrl else "vanilla"

    wandb.init(
        project="sac-reward-aug",
        name=f"{cfg.agent.id}_{cfg.env.id}_{rew_aug}",
        config=dict(cfg),
        reinit=True
    )

    env = make_env(cfg.env.id,
                   render_mode=cfg.env.render_mode,
                   max_episode_steps=cfg.env.max_episode_steps)

    agent = create_agent(cfg, env)

    start_time = time.time()
    agent.train(
        total_timesteps=cfg.agent.num_env_steps,
        max_episode_steps=cfg.env.max_episode_steps
    )
    elapsed = time.time() - start_time
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    log.info(f"[✓] Training finished in {int(h)}h {int(m)}m {int(s)}s")


if __name__ == "__main__":
    main()
