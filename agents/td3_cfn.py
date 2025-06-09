from pathlib import Path
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from cpprb import HindsightReplayBuffer, ReplayBuffer
from hydra.core.hydra_config import HydraConfig
import gymnasium as gym

from CFN.CFN import CoinFlipNetwork
from CFN.cfn_buffer import CFNReplayBufferWrapper
from CFN.priority_util import compute_intrinsic_reward, get_coin_flips
from agents.td3 import TD3Agent
from replay_buffer.replay_buffer import make_replay_buffer
from utils.evaluate import evaluate_cfn_bonus_generalization, evaluate
from utils.gif import save_rollout_gif
from utils.plots import plot_and_save_training_metrics, plot_eval_curve, cfn_early_vs_late_training_comparison
from utils.stats import EpisodeStats
import logging

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


class TD3CFNAgent(TD3Agent):
    def __init__(self, env, cfn_cfg, learning_rate=3e-4, buffer_size=int(1e6), gamma=0.99, tau=0.005, batch_size=256,
                 exploration_noise=0.1, learning_starts=25e3, policy_frequency=2, noise_clip=0.5, eval_env=None,
                 **kwargs):
        super().__init__(
            env=env,
            eval_env=eval_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            gamma=gamma,
            tau=tau,
            batch_size=batch_size,
            exploration_noise=exploration_noise,
            learning_starts=learning_starts,
            policy_frequency=policy_frequency,
            noise_clip=noise_clip,
            **kwargs
        )
        self.env = env
        self.eval_env = eval_env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
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
                "ext_rew": {},
                "int_rew": {},
                "next_obs": {"shape": obs_dim},
                "done": {}
            }
        )

        # Evaluation tracking
        self.eval_envsteps = []
        self.eval_means = []
        self.eval_stds = []

        # CFN components
        self.cfn_cfg = cfn_cfg
        self.use_cfn_prior = cfn_cfg.use_cfn_prior
        self.use_cfn_priority = cfn_cfg.use_cfn_priority
        self.coin_flip_dim = cfn_cfg.cfn_coin_flip_dim

        self.cfn = CoinFlipNetwork(obs_dim, self.coin_flip_dim)
        self.cfn_optimizer = optim.Adam(self.cfn.parameters(), lr=self.cfn_cfg.cfn_lr)

        self.cfn_buffer = CFNReplayBufferWrapper(
            size=cfn_cfg.cfn_replay_buffer_size,
            obs_shape=obs_dim,
            coin_flip_dim=self.coin_flip_dim,
            alpha=0.5
        )

    def train(self, total_timesteps, max_episode_steps):
        # CleanRL training loop with CFN integration
        start_time = time.time()
        obs, _ = self.env.reset()

        episode_lengths = []
        episode_rewards = []
        timesteps_on_ep_end = []
        episode_num = 0
        episode_reward = 0
        episode_length = 0

        for global_step in range(total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < self.learning_starts:
                actions = np.array([self.env.action_space.sample()])
            else:
                with torch.no_grad():
                    actions = self.actor(torch.Tensor(obs).unsqueeze(0))
                    actions += torch.normal(0, self.actor.action_scale * self.exploration_noise)
                    actions = actions.cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = self.env.step(actions[0])

            # CFN intrinsic reward computation
            intrinsic_reward = compute_intrinsic_reward(
                self.coin_flip_dim,
                self.cfn.compute_squared_output_norm(torch.as_tensor(obs).float())
            )

            if self.use_cfn_prior:
                self.cfn(torch.as_tensor(obs).float(), update_prior_stats=True)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            episode_reward += rewards + intrinsic_reward
            episode_length += 1

            if global_step % 1000 == 0:
                wandb.log({
                    "ext_reward": rewards,
                    "int_reward": intrinsic_reward,
                })

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            if truncations:
                real_next_obs = infos.get("final_observation", next_obs)

            # Store in main replay buffer
            self.buffer.add(
                obs=np.array(obs, dtype=np.float32),
                act=np.array(actions[0], dtype=np.float32),
                ext_rew=np.array(rewards, dtype=np.float32),
                int_rew=np.array(intrinsic_reward, dtype=np.float32),
                next_obs=np.array(real_next_obs, dtype=np.float32),
                done=np.array(terminations, dtype=np.float32)
            )

            # Store in CFN buffer
            coin_flip = get_coin_flips(self.coin_flip_dim)

            self.cfn_buffer.add(
                obs=np.array(obs, dtype=np.float32),
                coin_flip=coin_flip.detach().numpy(),
                priority=1.0
            )

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > self.learning_starts:
                if self.buffer.get_stored_size() >= self.batch_size:
                    data = self.buffer.sample(self.batch_size)

                    # Convert to tensors
                    observations = torch.FloatTensor(data["obs"])
                    actions = torch.FloatTensor(data["act"])
                    next_observations = torch.FloatTensor(data["next_obs"])
                    dones = torch.FloatTensor(data["done"])
                    ext_rewards = torch.FloatTensor(data["ext_rew"]).flatten()
                    int_rewards = torch.FloatTensor(data["int_rew"]).flatten()
                    rewards_combined = ext_rewards + int_rewards

                    with torch.no_grad():
                        clipped_noise = (torch.randn_like(actions) * self.noise_clip).clamp(
                            -self.noise_clip, self.noise_clip
                        ) * self.actor.action_scale

                        next_state_actions = (self.target_actor(next_observations) + clipped_noise).clamp(
                            torch.tensor(self.env.action_space.low), torch.tensor(self.env.action_space.high)
                        )

                        qf1_next_target = self.qf1_target(next_observations, next_state_actions)
                        qf2_next_target = self.qf2_target(next_observations, next_state_actions)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                        next_q_value = (
                                rewards_combined.squeeze(-1) +
                                (1 - dones.squeeze(-1)) * self.gamma * min_qf_next_target.squeeze(-1)
                        )

                    qf1_a_values = self.qf1(observations, actions).view(-1)
                    qf2_a_values = self.qf2(observations, actions).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    # optimize the model
                    self.q_optimizer.zero_grad()
                    qf_loss.backward()
                    self.q_optimizer.step()

                    if global_step % self.policy_frequency == 0:  # TD 3 Delayed update support
                        actor_loss = -self.qf1(observations, self.actor(observations)).mean()
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor_optimizer.step()

                        # update the target network
                        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                        if global_step % 1000 == 0:
                            wandb.log({
                                "losses/qf1_loss": qf1_loss.item(),
                                "losses/qf2_loss": qf2_loss.item(),
                                "losses/qf_loss": qf_loss.item(),
                                "losses/actor_loss": actor_loss.item(),
                            }, step=global_step)

                # Update CFN
                if self.cfn_buffer.get_stored_size() >= self.cfn_cfg.cfn_batch_size:
                    obs_batch_bc, coin_flip_batch_bc, _ = self.cfn_buffer.sample_and_update_priorities(
                        batch_size=self.cfn_cfg.cfn_batch_size,
                        cfn=self.cfn,
                        coin_flip_dim=self.coin_flip_dim,
                        use_cfn_priority=self.use_cfn_priority
                    )
                    self.update_cfn(obs_batch_bc, coin_flip_batch_bc)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if terminations or truncations or episode_length >= max_episode_steps:
                episode_lengths.append(episode_length)
                episode_rewards.append(episode_reward)
                timesteps_on_ep_end.append(global_step)

                wandb.log({
                    "charts/episodic_return": episode_reward,
                    "charts/episodic_length": episode_length,
                    "charts/episode_num": episode_num,
                }, step=global_step)

                log.info(
                    f"Episode {episode_num} | Steps: {episode_length} | "
                    f"Return: {episode_reward:.2f} | Total Timesteps: {global_step}"
                )

                # Reset environment
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_num += 1

            # Evaluation
            if global_step % 1000 == 0:
                validate(self.actor, global_step)
                evaluate(self.actor, self.eval_env, current_timestep=global_step, max_steps=max_episode_steps)

        # Final evaluation and plotting
        stats = EpisodeStats(
            episode_lengths=episode_lengths,
            episode_rewards=episode_rewards,
            timesteps_on_ep_end=timesteps_on_ep_end
        )

        plot_and_save_training_metrics(
            stats,
            output_dir=Path(HydraConfig.get().runtime.output_dir),
            tag="td3_run"
        )

        plot_path = Path(HydraConfig.get().runtime.output_dir) / "validate" / f"eval_plot_step{total_timesteps}.png"
        plot_eval_curve(self.eval_envsteps, self.eval_means, self.eval_stds, plot_path)

        save_rollout_gif(self.actor, self.env, Path(
            HydraConfig.get().runtime.output_dir) / "validate" / f"eval_gif{total_timesteps}.gif")

        cfn_early_vs_late_training_comparison(self.cfn,
                                              eval_dir=Path(HydraConfig.get().runtime.output_dir) / "evaluate")

        evaluate_cfn_bonus_generalization(self.cfn, self.env, self.buffer)

    def update_cfn(self, obs_batch: torch.Tensor, coin_flip_batch: torch.Tensor):
        """
        Update the Coin Flip Network with the current batch of states.
        """
        predicted_coin_flips = self.cfn(obs_batch)
        cfn_loss = F.mse_loss(predicted_coin_flips, coin_flip_batch)

        self.cfn_optimizer.zero_grad()
        cfn_loss.backward()
        self.cfn_optimizer.step()
