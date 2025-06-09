import logging

import hydra
import torch
import numpy as np
import os
from pathlib import Path
import re
import wandb
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import TimeLimit

from utils.env_wrapper import make_env

gym.register_envs(gymnasium_robotics)
from hydra.core.hydra_config import HydraConfig

from networks.network import Actor
from utils.evaluate import evaluate
from utils.gif import save_rollout_gif

log = logging.getLogger(__name__)



def load_actor(actor_path, actor_class, obs_dim, act_dim, act_low, act_high):
    actor = actor_class(obs_dim, act_dim, act_low, act_high)
    actor.load_state_dict(torch.load(actor_path, map_location=torch.device('cpu')))
    actor.eval()
    return actor


# TODO:make this config as conf-evaluate
# @hydra.main(config_path="conf", config_name="config", version_base=None)
def main():
    wandb.init(
        project="sac-reward-aug-evaluate",
        reinit=True
    )

    # TODO: will go in a config
    actor_path = "outputs/2025-06-09/09-51-28-sac_agent-FetchReachDense-v3-True/checkpoints/sac_actor_step300000.pt"

    max_steps = 1000

    # TODO: put it in a make_env()
    env = make_env(
        env_name="FetchReachDense-v3",
        render_mode="rgb_array",
        max_episode_steps=100,
    )

    actor = load_actor(actor_path, Actor,
                       env.observation_space.shape[0],
                       env.action_space.shape[0],
                       env.action_space.low,
                       env.action_space.high)

    # TODO: add an output dir param
    _, mean_r, std_r = evaluate(actor, env, 0, max_steps, "evaluate_any_outputs")
    log.info(f"evaluation of {actor_path}")
    log.info(f"mean return: {mean_r} ± {std_r}")

    #     TODO: add plots and gifs
    save_rollout_gif(actor, env, "evaluate_any_outputs/eval_gif.gif")


if __name__ == "__main__":
    main()
