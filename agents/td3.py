from pathlib import Path
import copy
import random
import time

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from cpprb import ReplayBuffer
from utils.stats import EpisodeStats
import logging
import gymnasium as gym

from utils.validate import validate

log = logging.getLogger(__name__)


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Handle both dict and box observation spaces
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
        else:
            obs_dim = np.array(env.observation_space.shape).prod()

        action_dim = np.prod(env.action_space.shape)

        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        # Handle dict observations by flattening
        if isinstance(x, dict):
            x = torch.cat([v.flatten(1) if v.dim() > 1 else v for v in x.values()], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Handle both dict and box observation spaces
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
        else:
            obs_dim = np.array(env.observation_space.shape).prod()

        action_dim = np.prod(env.action_space.shape)

        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        # Handle dict observations by flattening
        if isinstance(x, dict):
            x = torch.cat([v.flatten(1) if v.dim() > 1 else v for v in x.values()], dim=1)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TD3Agent:
    def __init__(
            self,
            env,
            learning_rate=1e-3,
            buffer_size=int(1e6),
            gamma=0.95,
            tau=0.05,
            batch_size=256,
            policy_noise=0.2,
            exploration_noise=0.1,
            learning_starts=10000,
            policy_frequency=2,
            noise_clip=0.5,
            **kwargs
    ):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.exploration_noise = exploration_noise
        self.learning_starts = int(learning_starts)
        self.policy_frequency = policy_frequency
        self.noise_clip = noise_clip

        self.actor = Actor(env)
        self.qf1 = QNetwork(env)
        self.qf2 = QNetwork(env)
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)
        self.target_actor = copy.deepcopy(self.actor)

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=learning_rate)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=learning_rate)

        # Determine observation dimension for replay buffer
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_dim = sum([np.prod(space.shape) for space in env.observation_space.spaces.values()])
        else:
            obs_dim = np.array(env.observation_space.shape).prod()

        # Replay buffer
        self.buffer = ReplayBuffer(
            size=buffer_size,
            env_dict={
                "obs": {"shape": obs_dim},
                "act": {"shape": env.action_space.shape[0]},
                "rew": {},
                "next_obs": {"shape": obs_dim},
                "done": {}
            }
        )

        # Evaluation tracking
        self.eval_envsteps = []
        self.eval_means = []
        self.eval_stds = []

    def _flatten_obs(self, obs):
        """Flatten observation if it's a dict, otherwise return as is"""
        if isinstance(obs, dict):
            return np.concatenate([v.flatten() for v in obs.values()])
        return obs

    def train(self, total_timesteps, max_episode_steps):
        start_time = time.time()
        obs, _ = self.env.reset()

        episode_lengths = []
        episode_rewards = []
        timesteps_on_ep_end = []
        episode_num = 0
        episode_reward = 0
        episode_length = 0

        for global_step in range(total_timesteps):
            # Action selection
            if global_step < self.learning_starts:
                actions = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(self._flatten_obs(obs)).unsqueeze(0)
                    actions = self.actor(obs_tensor).cpu().numpy()[0]
                    actions += np.random.normal(0, self.exploration_noise * (
                            self.env.action_space.high - self.env.action_space.low) / 2)
                    actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

            # Execute action
            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

            episode_reward += rewards
            episode_length += 1

            # Handle final observation for truncated episodes
            real_next_obs = next_obs.copy()
            if truncations:
                real_next_obs = infos.get("final_observation", next_obs)

            # Store in replay buffer
            self.buffer.add(
                obs=self._flatten_obs(obs).astype(np.float32),
                act=actions.astype(np.float32),
                rew=np.array(rewards, dtype=np.float32),
                next_obs=self._flatten_obs(real_next_obs).astype(np.float32),
                done=np.array(terminations, dtype=np.float32)
            )

            obs = next_obs

            # Training
            if global_step > self.learning_starts and self.buffer.get_stored_size() >= self.batch_size:
                data = self.buffer.sample(self.batch_size)

                # Convert to tensors
                observations = torch.FloatTensor(data["obs"])
                actions = torch.FloatTensor(data["act"])
                next_observations = torch.FloatTensor(data["next_obs"])
                dones = torch.FloatTensor(data["done"])
                rewards = torch.FloatTensor(data["rew"]).flatten()

                with torch.no_grad():
                    # Target policy smoothing
                    clipped_noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                        -self.noise_clip, self.noise_clip
                    )
                    next_state_actions = (self.target_actor(next_observations) + clipped_noise).clamp(
                        torch.tensor(self.env.action_space.low),
                        torch.tensor(self.env.action_space.high)
                    )

                    # Compute target Q-values
                    qf1_next_target = self.qf1_target(next_observations, next_state_actions)
                    qf2_next_target = self.qf2_target(next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    next_q_value = rewards + (1 - dones) * self.gamma * min_qf_next_target.squeeze(-1)

                # Current Q-value estimates
                qf1_a_values = self.qf1(observations, actions).view(-1)
                qf2_a_values = self.qf2(observations, actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # Update critics
                self.q_optimizer.zero_grad()
                qf_loss.backward()
                self.q_optimizer.step()

                # Delayed policy updates
                if global_step % self.policy_frequency == 0:
                    # Update actor
                    actor_loss = -self.qf1(observations, self.actor(observations)).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Update target networks
                    for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                        target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                    # Logging
                    if global_step % 100 == 0:
                        wandb.log({
                            "losses/qf1_loss": qf1_loss.item(),
                            "losses/qf2_loss": qf2_loss.item(),
                            "losses/qf_loss": qf_loss.item(),
                            "losses/actor_loss": actor_loss.item()
                        }, step=global_step)

            # Episode end handling
            if terminations or truncations or episode_length >= max_episode_steps:
                episode_lengths.append(episode_length)
                episode_rewards.append(episode_reward)
                timesteps_on_ep_end.append(global_step)

                log.info(
                    f"Episode {episode_num} | Steps: {episode_length} | "
                    f"Return: {episode_reward:.2f} | Total Timesteps: {global_step}"
                )

                wandb.log({
                    "charts/episodic_return": episode_reward,
                    "charts/episodic_length": episode_length,
                    "charts/episode_num": episode_num,
                }, step=global_step)

                # Reset environment
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_num += 1

            # Periodic evaluation
            if global_step % 5000 == 0 and global_step > 0:
                validate(self.actor, global_step)
                self.evaluate(self.actor, self.env, total_timesteps)
                # eval_envstep, eval_mean, eval_std = evaluate(self.actor, self.env, global_step, max_episode_steps)
                # self.eval_envsteps.append(eval_envstep)
                # self.eval_means.append(eval_mean)
                # self.eval_stds.append(eval_std)

        # Final evaluation and plotting
        stats = EpisodeStats(
            episode_lengths=episode_lengths,
            episode_rewards=episode_rewards,
            timesteps_on_ep_end=timesteps_on_ep_end
        )

        # self.save_rollout_gif(self.actor, self.env, "rollout.gif")

        return stats

    def evaluate(self, actor, env, step, max_episode_steps=1000, num_episodes=10, render=False):
        actor.eval()
        returns = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            episode_length = 0

            while not done and episode_length < max_episode_steps:
                obs_tensor = torch.FloatTensor(obs if not isinstance(obs, dict) else
                                               np.concatenate([v.flatten() for v in obs.values()])).unsqueeze(0)
                with torch.no_grad():
                    action = actor(obs_tensor).cpu().numpy()[0]
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                episode_length += 1

                if render:
                    env.render()

            returns.append(total_reward)

        actor.train()
        eval_mean = np.mean(returns)
        eval_std = np.std(returns)
        print(f"[Evaluation @ Step {step}] Mean: {eval_mean:.2f} | Std: {eval_std:.2f}")

        # WandB logging
        wandb.log({"eval/return_mean": eval_mean, "eval/return_std": eval_std, "global_step": step})

        return step, eval_mean, eval_std

    def save_rollout_gif(self, actor, env, gif_path, max_episode_steps=1000):
        actor.eval()
        frames = []

        obs, _ = env.reset()
        done = False
        step = 0

        while not done and step < max_episode_steps:
            frame = env.render(mode="rgb_array")
            frames.append(frame)

            obs_tensor = torch.FloatTensor(obs if not isinstance(obs, dict) else
                                           np.concatenate([v.flatten() for v in obs.values()])).unsqueeze(0)
            with torch.no_grad():
                action = actor(obs_tensor).cpu().numpy()[0]
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1

        actor.train()

        imageio.mimsave(gif_path, frames, fps=30)
    #
