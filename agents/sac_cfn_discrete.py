import copy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from pathlib import Path
import wandb
import logging
from cpprb import ReplayBuffer
from hydra.core.hydra_config import HydraConfig

from CFN.CFN import CoinFlipNetwork
from CFN.cfn_buffer import CFNReplayBufferWrapper
from CFN.priority_util import compute_intrinsic_reward, get_coin_flips
from utils.evaluate import evaluate_cfn_bonus_generalization, evaluate
from utils.gif import save_rollout_gif
from utils.plots import plot_and_save_training_metrics, plot_eval_curve, cfn_early_vs_late_training_comparison
from utils.stats import EpisodeStats
from utils.validate import validate

log = logging.getLogger(__name__)


class CNNEncoder(nn.Module):
    def __init__(self, input_shape, output_dim=256):
        super().__init__()
        c, h, w = input_shape
        assert h >= 5 and w >= 5, f"Expected at least 5x5 input, got {h}x{w}"

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1),  # -> [B, 32, H-2, W-2]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # -> [B, 64, H-4, W-4]
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            conv_out_dim = self.conv(dummy_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class DiscreteActor(nn.Module):
    def __init__(self, encoder, action_dim):
        super().__init__()
        self.encoder = encoder
        self.policy_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, obs):
        x = self.encoder(obs)
        logits = self.policy_head(x)
        return logits



class Critic(nn.Module):
    def __init__(self, encoder, action_dim, encoder_output_dim):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.action_dim = action_dim

        self.fc = nn.Sequential(
            nn.Linear(encoder_output_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, action):
        x = self.encoder(obs)
        if action.ndim == 1:
            action = F.one_hot(action, num_classes=self.action_dim).float()
        x = torch.cat([x, action], dim=-1)
        return self.fc(x)


class SACCFNAgentDiscrete:
    def __init__(self, env, cfn_cfg, lr, gamma, tau, batch_size, maxlen, target_entropy, eval_env):
        self.env = env
        self.eval_env = eval_env
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = 0.2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_shape = env.observation_space.shape
        self.action_dim = env.action_space.n
        self.encoder = CNNEncoder(self.obs_shape).to(self.device)
        self.actor = DiscreteActor(self.encoder, self.action_dim).to(self.device)

        # Determine encoder output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.obs_shape).to(self.device)
            encoder_output_dim = self.encoder(dummy_input).shape[1]

        self.q1 = Critic(self.encoder, self.action_dim, encoder_output_dim).to(self.device)
        self.q2 = Critic(self.encoder, self.action_dim, encoder_output_dim).to(self.device)
        self.q1_target = Critic(self.encoder, self.action_dim, encoder_output_dim).to(self.device)
        self.q2_target = Critic(self.encoder, self.action_dim, encoder_output_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=self.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.buffer = ReplayBuffer(
            size=100_000,
            env_dict={
                "obs": {"shape": self.obs_shape},
                "act": {},
                "ext_rew": {},
                "int_rew": {},
                "next_obs": {"shape": self.obs_shape},
                "done": {},
            }
        )

        self.cfn_cfg = cfn_cfg
        self.use_cfn_prior = cfn_cfg.use_cfn_prior
        self.use_cfn_priority = cfn_cfg.use_cfn_priority
        self.coin_flip_dim = cfn_cfg.cfn_coin_flip_dim

        

        # Correctly initialize the CFN with encoder output dimension
        self.cfn = CoinFlipNetwork(encoder_output_dim, self.coin_flip_dim).to(self.device)
        self.cfn_optimizer = torch.optim.Adam(self.cfn.parameters(), lr=self.cfn_cfg.cfn_lr)

        self.cfn_buffer = CFNReplayBufferWrapper(
            size=cfn_cfg.cfn_replay_buffer_size,
            obs_shape=encoder_output_dim,
            coin_flip_dim=self.coin_flip_dim,
            alpha=0.5
        )

        self.is_goal_env = hasattr(env, 'is_goal_env') and env.is_goal_env
        self.eval_envsteps = []
        self.eval_means = []
        self.eval_stds = []


    def act(self, obs, deterministic=False):
        logits = self.actor(obs.unsqueeze(0))
        dist = Categorical(logits=logits)
        return torch.argmax(logits, dim=-1) if deterministic else dist.sample()

    def update(self, obs, act, rew, next_obs, done, step):
        with torch.no_grad():
            logits_next = self.actor(next_obs)
            dist_next = Categorical(logits=logits_next)
            probs = dist_next.probs

            all_actions = torch.eye(self.action_dim, device=self.device).unsqueeze(0).repeat(next_obs.shape[0], 1, 1)
            next_obs_repeat = next_obs.unsqueeze(1).repeat(1, self.action_dim, 1, 1, 1)
            q1_vals = self.q1_target(next_obs_repeat.view(-1, *next_obs.shape[1:]), all_actions.view(-1, self.action_dim))
            q2_vals = self.q2_target(next_obs_repeat.view(-1, *next_obs.shape[1:]), all_actions.view(-1, self.action_dim))
            min_q = torch.min(q1_vals, q2_vals).view(next_obs.shape[0], self.action_dim)

            entropy = dist_next.entropy()
            target_q = (probs * (min_q - self.alpha * torch.log(probs + 1e-8))).sum(dim=1)
            target_q = rew + self.gamma * (1 - done) * target_q

        q1_pred = self.q1(obs, act).squeeze(-1)
        q2_pred = self.q2(obs, act).squeeze(-1)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        probs = dist.probs

        all_actions = torch.eye(self.action_dim, device=self.device).unsqueeze(0).repeat(obs.shape[0], 1, 1)
        obs_repeat = obs.unsqueeze(1).repeat(1, self.action_dim, 1, 1, 1)
        q1_vals = self.q1(obs_repeat.view(-1, *obs.shape[1:]), all_actions.view(-1, self.action_dim))
        q1_vals = q1_vals.view(obs.shape[0], self.action_dim)

        actor_loss = (probs * (self.alpha * torch.log(probs + 1e-8) - q1_vals)).sum(dim=1).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_cfn(self, obs_batch: torch.Tensor, coin_flip_batch: torch.Tensor):
        predicted_coin_flips = self.cfn(obs_batch)
        cfn_loss = F.mse_loss(predicted_coin_flips, coin_flip_batch)
        self.cfn_optimizer.zero_grad()
        cfn_loss.backward()
        self.cfn_optimizer.step()

    def train(self, total_timesteps, max_episode_steps, learning_starts=10000):
        current_timestep = 0
        episode_return = 0
        episode_step = 0
        episode_num = 0
        episode_lengths = []
        episode_rewards = []
        timesteps_on_ep_end = []
        avg_rewards = []

        obs_np, _ = self.env.reset()
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        while current_timestep < total_timesteps:
            with torch.no_grad():
                logits = self.actor(obs)
                dist = Categorical(logits=logits)
                action = dist.sample().item()

            next_obs_np, reward, terminated, truncated, info = self.env.step(action)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            done = terminated or truncated

            encoded_obs = self.encoder(obs)[0]
            intrinsic_reward = compute_intrinsic_reward(
                self.coin_flip_dim,
                self.cfn.compute_squared_output_norm(encoded_obs)
            )

            avg_rewards.append(intrinsic_reward.item() + reward)

            if self.use_cfn_prior:
                with torch.no_grad():
                    cfn_input = encoded_obs
                    combined_out = self.cfn(cfn_input, update_prior_stats=False)
                    prior_out = self.cfn.prior(cfn_input)
                    output_norm = combined_out.norm(p=2, dim=0)
                    prior_output_norm = prior_out.norm(p=2, dim=0)
                    pseudocount_estimate = self.cfn.coin_flip_dim / (output_norm ** 2)

                    wandb.log({
                        "pseudocounts-intr": 1 / (intrinsic_reward ** 2 + 1e-8).item(),
                        "prior_output_norm": prior_output_norm.item(),
                        "output_norm": output_norm.item(),
                        "pseudocount_estimate": pseudocount_estimate.item(),
                    }, step=current_timestep)

                self.cfn(cfn_input, update_prior_stats=True)

            if current_timestep % 1000 == 0:
                wandb.log({
                    "ext_reward": reward,
                    "int_reward": intrinsic_reward.item(),
                    "averaged_rewards": np.average(avg_rewards),
                }, step=current_timestep)
                avg_rewards.clear()

            self.buffer.add(
                obs=obs.squeeze(0).cpu().numpy(),
                act=action,
                ext_rew=reward,
                int_rew=intrinsic_reward.item(),
                next_obs=next_obs.squeeze(0).cpu().numpy(),
                done=float(done)
            )

            coin_flip = get_coin_flips(self.coin_flip_dim)
            self.cfn_buffer.add(
                obs=encoded_obs.detach().cpu().numpy(),
                coin_flip=coin_flip.detach().cpu().numpy(),
                priority=1.0
            )

            if current_timestep >= learning_starts:
                sample = self.buffer.sample(self.batch_size)

                obs_batch = torch.tensor(sample["obs"], dtype=torch.float32, device=self.device)
                next_obs_batch = torch.tensor(sample["next_obs"], dtype=torch.float32, device=self.device)
                act_batch = torch.tensor(sample["act"], dtype=torch.int64, device=self.device)
                ext_rew_batch = torch.tensor(sample["ext_rew"], dtype=torch.float32, device=self.device)
                int_rew_batch = torch.tensor(sample["int_rew"], dtype=torch.float32, device=self.device)
                tm_batch = torch.tensor(sample["done"], dtype=torch.float32, device=self.device)

                total_rew_batch = int_rew_batch + ext_rew_batch
                self.update(obs_batch, act_batch, total_rew_batch, next_obs_batch, tm_batch, current_timestep)

            if self.cfn_buffer.size >= self.cfn_cfg.cfn_batch_size:
                obs_batch_bc, coin_flip_batch_bc, indices = self.cfn_buffer.sample_with_indices(
                    batch_size=self.cfn_cfg.cfn_batch_size
                )
                self.update_cfn(obs_batch_bc, coin_flip_batch_bc)

                if self.use_cfn_priority:
                    self.cfn_buffer.update_priorities(
                        indices, obs_batch_bc, self.cfn, self.coin_flip_dim
                    )

            obs = next_obs
            episode_return += reward + intrinsic_reward.item()
            episode_step += 1
            current_timestep += 1

            if done or episode_step >= max_episode_steps:
                episode_lengths.append(episode_step)
                episode_rewards.append(episode_return)
                timesteps_on_ep_end.append(current_timestep)

                wandb.log({
                    "charts/episodic_return": episode_return,
                    "charts/episodic_length": episode_step,
                    "charts/episode_num": episode_num
                }, step=current_timestep)

                log.info(
                    f"Episode {episode_num} | Steps: {episode_step} | "
                    f"Return: {episode_return:.2f} | Total Timesteps: {current_timestep}"
                )

                obs_np, _ = self.env.reset()
                obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)

                episode_return = 0
                episode_step = 0
                episode_num += 1

            # if current_timestep % 10000 == 0:
            #     validate(self.actor, current_timestep)
            #     eval_envstep, eval_mean, eval_std = evaluate(self.actor, self.eval_env, current_timestep, max_episode_steps)
            #     self.eval_envsteps.append(eval_envstep)
            #     self.eval_means.append(eval_mean)
            #     self.eval_stds.append(eval_std)

        stats = EpisodeStats(
            episode_lengths=episode_lengths,
            episode_rewards=episode_rewards,
            timesteps_on_ep_end=timesteps_on_ep_end
        )

        plot_and_save_training_metrics(
            stats,
            output_dir=Path(HydraConfig.get().runtime.output_dir),
            tag="sac_run"
        )

        plot_path = Path(HydraConfig.get().runtime.output_dir) / "validate" / f"eval_plot_step{current_timestep}.png"
        plot_eval_curve(self.eval_envsteps, self.eval_means, self.eval_stds, plot_path)

        cfn_early_vs_late_training_comparison(self.cfn,
                                              eval_dir=Path(HydraConfig.get().runtime.output_dir) / "evaluate")

        evaluate_cfn_bonus_generalization(self.cfn, self.env, self.buffer)
