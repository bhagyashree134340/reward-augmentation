import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
import random
import gymnasium as gym
from collections import deque
from pathlib import Path
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from networks.ddqn import DDQN
from utils.validate import validate_dqn
from utils.evaluate import evaluate_dqn
from utils.stats import EpisodeStats
import logging

log = logging.getLogger(__name__)

class DQNAgent:
    def __init__(self, env, eval_env, env_name, cfg):
        self.env = env
        self.eval_env = eval_env
        self.env_name = env_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_shape = env.observation_space.shape
        self.obs_shape_torch = (self.obs_shape[2], self.obs_shape[0], self.obs_shape[1])
        self.act_dim = env.action_space.n

        self.q_net = DDQN(self.obs_shape_torch, self.act_dim, cfg.hidden_size, is_cnn=True).to(self.device)
        self.target_q_net = DDQN(self.obs_shape_torch, self.act_dim, cfg.hidden_size, is_cnn=True).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.target_update_freq = cfg.target_update_freq
        self.replay_buffer = deque(maxlen=cfg.replay_buffer_size)

    def process_obs(self, obs):
        if isinstance(obs, dict) and 'image' in obs:
            obs = obs['image']
        return np.transpose(np.array(obs, dtype=np.uint8), (2, 0, 1))

    def act(self, obs, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return q_values.argmax().item()

    def update(self, batch, current_timestep, update_count):
        obs, act, rew, next_obs, done = map(np.array, zip(*batch))
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        act = torch.tensor(act, dtype=torch.long, device=self.device)
        rew = torch.tensor(rew, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        q_vals = self.q_net(obs)
        q_val = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_obs).argmax(1)
            next_q_vals_target = self.target_q_net(next_obs)
            max_next_q_vals = next_q_vals_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rew + (1 - done) * self.gamma * max_next_q_vals

        loss = F.mse_loss(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if update_count % 1000 == 0:
            wandb.log({
                "training/dqn_loss": loss.item(),
                "training/q_values_mean": q_val.mean().item(),
                "training/target_mean": target.mean().item(),
            }, step=current_timestep)

    def train(self, total_timesteps, max_episode_steps, epsilon_start, epsilon_end, epsilon_decay):
        current_timestep = 0
        episode_num = 0
        episode_return = 0
        episode_step = 0
        update_count = 0

        obs_raw, _ = self.env.reset()
        obs = self.process_obs(obs_raw)

        stats = EpisodeStats([], [], [])

        epsilon = epsilon_start
        learning_starts = max(10000, self.batch_size * 4)

        while current_timestep < total_timesteps:
            action = self.act(obs, epsilon)
            next_obs_raw, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = self.process_obs(next_obs_raw)

            done = terminated or truncated

            if reward>0:
                wandb.log({
                    "reward": reward
                }, step = current_timestep)

            self.replay_buffer.append((obs, action, reward, next_obs, done))

            if current_timestep >= learning_starts and len(self.replay_buffer) >= self.batch_size:
                batch = random.sample(self.replay_buffer, self.batch_size)
                self.update(batch, current_timestep, update_count)
                update_count += 1

            if current_timestep % self.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            obs = next_obs
            episode_return += reward
            episode_step += 1
            current_timestep += 1

            if done:
                stats.episode_rewards.append(episode_return)
                stats.episode_lengths.append(episode_step)
                stats.timesteps_on_ep_end.append(current_timestep)

                wandb.log({
                    "charts/episodic_return": episode_return,
                    "charts/episodic_length": episode_step,
                    "charts/episode_num": episode_num
                }, step=current_timestep)

                print(
                    f"Episode {episode_num} | Steps: {episode_step} | "
                    f"Return: {reward:.2f} | Epsilon: {epsilon:.3f} | "
                    f"Total Timesteps: {current_timestep}"
                )

                if epsilon > epsilon_end:
                    epsilon *= epsilon_decay

                obs_raw, _ = self.env.reset()
                obs = self.process_obs(obs_raw)
                episode_return = 0
                episode_step = 0
                episode_num += 1

            if current_timestep % 10000 == 0 and current_timestep > 0:
                validate_dqn(self, current_timestep)
                evaluate_dqn(self, self.env, current_timestep)

def main():
    ENV_NAME = "MiniGrid-DoorKey-6x6-v0"
    wandb.init(project="dqn", name="vanilla-dqn")

    env = gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=250)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    eval_env = gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=250)
    eval_env = FullyObsWrapper(eval_env)
    eval_env = ImgObsWrapper(eval_env)

    cfg = type("DQNConfig", (), {
        "hidden_size": 256,
        "lr": 1e-5,
        "gamma": 0.99,
        "batch_size": 128,
        "replay_buffer_size": 1_000_000,
        "target_update_freq": 2000,
    })

    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.9998

    agent = DQNAgent(env, eval_env, ENV_NAME, cfg)

    start = time.time()
    agent.train(
        total_timesteps=1_000_000,
        max_episode_steps=250,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay
    )
    end = time.time()
    print(f"Training finished in {(end - start) / 60:.2f} minutes.")

if __name__ == "__main__":
    main()