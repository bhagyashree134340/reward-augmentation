from pathlib import Path

import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
import torch.nn.functional as F
from agents.sac_agent import SACAgent
from utils.evaluate import evaluate
from utils.plots import plot_and_save_training_metrics, plot_eval_curve
from utils.stats import EpisodeStats
import logging
from gym.wrappers.normalize import RunningMeanStd
from torch import nn
from utils.validate import validate

log = logging.getLogger(__name__)


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size=512):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        self.target = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        pred = self.predictor(x)
        target = self.target(x).detach()
        return pred, target


class SACRNDAgent(SACAgent):
    def __init__(self, env, rnd_cfg, **kwargs):
        super().__init__(env, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_rnd = True  # TODO: change this to take from config
        self.update_proportion = 0.25
        self.rnd = RNDModel(env.observation_space.shape[0]).to(self.device)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=1e-4)
        self.obs_rms = RunningMeanStd(shape=env.observation_space.shape)
        self.int_reward_rms = RunningMeanStd(shape=())

        self.is_goal_env = hasattr(env, 'is_goal_env') and env.is_goal_env

    def train(self, total_timesteps, max_episode_steps, learning_starts=10000):
        current_timestep = 0
        episode_return = 0
        episode_step = 0
        episode_num = 0
        episode_lengths = []
        episode_rewards = []
        timesteps_on_ep_end = []

        obs_np, _ = self.env.reset()
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)

        # ---- RND observation normalization warm-up START----
        M = 1000
        for _ in range(M):
            random_action = self.env.action_space.sample()
            next_obs_np, _, terminated, truncated, _ = self.env.step(random_action)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)
            self.obs_rms.update(next_obs[None, :].cpu().numpy())

            done = terminated or truncated
            if done:
                obs_np, _ = self.env.reset()
                obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
            else:
                obs = next_obs
        # ---- RND observation normalization warm-up END----

        reward_forward_filter = RewardForwardFilter(gamma=0.99)  # TODO: Config

        while current_timestep < total_timesteps:
            gripper_pos = obs[:3]
            goal_pos = obs[-3:]
            distance = np.linalg.norm(gripper_pos.cpu().numpy() - goal_pos.cpu().numpy())
            wandb.log({"gripper-goal-dist": distance}, step=current_timestep)

            with torch.no_grad():
                action, _ = self.actor(obs)
                action = action.cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)

            next_obs_np, reward, terminated, truncated, info = self.env.step(action)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)
            done = terminated or truncated

            self.obs_rms.update(next_obs[None, :].cpu().numpy())
            mean = torch.tensor(self.obs_rms.mean, device=self.device, dtype=next_obs.dtype)
            var = torch.tensor(self.obs_rms.var, device=self.device, dtype=next_obs.dtype)
            next_obs_normed = (next_obs - mean) / torch.sqrt(var + 1e-8)
            next_obs_normed = torch.clamp(next_obs_normed, -5.0, 5.0)

            obs_tensor = next_obs_normed.unsqueeze(0)

            with torch.no_grad():
                predict_feature, target_feature = self.rnd(obs_tensor)
                raw_int_reward = ((target_feature - predict_feature).pow(2).sum(1) / 2).data

            filtered_int_reward = reward_forward_filter.update(np.array([raw_int_reward.cpu().item()]))
            self.int_reward_rms.update_from_moments(
                np.mean(filtered_int_reward), np.var(filtered_int_reward), len(filtered_int_reward)
            )
            normed_int_reward = raw_int_reward / np.sqrt(self.int_reward_rms.var + 1e-8)

            self.buffer.add(
                obs=obs.detach().cpu().numpy(),
                act=action,
                ext_rew=np.array(reward, dtype=np.float32),
                int_rew=np.array(normed_int_reward.item(), dtype=np.float32),
                next_obs=next_obs.detach().cpu().numpy(),
                done=np.array(done, dtype=np.float32)
            )

            if current_timestep >= learning_starts:
                sample = self.buffer.sample(self.batch_size)

                obs_batch = torch.tensor(sample["obs"], dtype=torch.float32, device=self.device)
                next_obs_batch = torch.tensor(sample["next_obs"], dtype=torch.float32, device=self.device)
                act_batch = torch.tensor(sample["act"], dtype=torch.float32, device=self.device)
                ext_rew_batch = torch.tensor(sample["ext_rew"], dtype=torch.float32, device=self.device).squeeze(-1)
                int_rew_batch = torch.tensor(sample["int_rew"], dtype=torch.float32, device=self.device).squeeze(-1)
                tm_batch = torch.tensor(sample["done"], dtype=torch.float32, device=self.device).squeeze(-1)

                total_rew_batch = int_rew_batch + ext_rew_batch
                self.update(obs_batch, act_batch, total_rew_batch, next_obs_batch, tm_batch, current_timestep)

                rnd_loss = self.update_rnd(next_obs_batch)
                wandb.log({"loss/rnd_loss": rnd_loss}, step=current_timestep)

            wandb.log({
                "rewards/extrinsic": reward,
                "rewards/intrinsic": normed_int_reward,
                "rewards/total": reward + normed_int_reward
            }, step=current_timestep)

            obs = next_obs
            episode_return += reward + normed_int_reward
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
                    f"Return: {episode_return.item():.2f} | Total Timesteps: {current_timestep}"
                )

                obs_np, _ = self.env.reset()
                obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
                episode_return = 0
                episode_step = 0
                episode_num += 1

            if current_timestep % 10000 == 0:
                validate(self.actor, current_timestep)
                eval_envstep, eval_mean, eval_std = evaluate(
                    self.actor, self.eval_env, current_timestep, max_episode_steps)
                self.eval_envsteps.append(eval_envstep)
                self.eval_means.append(eval_mean)
                self.eval_stds.append(eval_std)

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

    def update_rnd(self, obs_batch: torch.Tensor) -> float:
        obs_normed = (obs_batch - torch.as_tensor(self.obs_rms.mean, device=self.device)) / \
                     torch.sqrt(torch.as_tensor(self.obs_rms.var, device=self.device) + 1e-8)
        obs_normed = torch.clamp(obs_normed, -5.0, 5.0).float()

        predict_feat, target_feat = self.rnd(obs_normed)
        forward_loss = F.mse_loss(predict_feat, target_feat, reduction='none').mean(dim=-1)

        mask = (torch.rand(len(forward_loss), device=self.device) < self.update_proportion).float()
        masked_loss = (forward_loss * mask).sum() / (mask.sum() + 1e-8)

        self.rnd_optimizer.zero_grad()
        masked_loss.backward()
        self.rnd_optimizer.step()

        return masked_loss.item()
