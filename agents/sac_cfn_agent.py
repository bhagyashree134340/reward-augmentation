from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import wandb
from cpprb import HindsightReplayBuffer
from hydra.core.hydra_config import HydraConfig
import torch.nn.functional as F
from CFN.CFN import CoinFlipNetwork
from CFN.cfn_buffer import CFNReplayBufferWrapper
from CFN.priority_util import compute_intrinsic_reward, get_coin_flips
from agents.sac_agent import SACAgent
from replay_buffer.replay_buffer import make_replay_buffer
from utils.evaluate import evaluate_cfn_bonus_generalization, evaluate
from utils.gif import save_rollout_gif
from utils.plots import log_cfn_stats_to_wandb, plot_and_save_training_metrics, plot_eval_curve, \
    cfn_early_vs_late_training_comparison
from utils.stats import EpisodeStats
import logging

from utils.validate import validate

log = logging.getLogger(__name__)


class SACCFNAgent(SACAgent):
    def __init__(self, env, cfn_cfg, **kwargs):
        super().__init__(env, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cfn_cfg = cfn_cfg
        self.use_cfn_prior = cfn_cfg.use_cfn_prior
        self.use_cfn_priority = cfn_cfg.use_cfn_priority
        self.coin_flip_dim = cfn_cfg.cfn_coin_flip_dim

        self.cfn = CoinFlipNetwork(env.observation_space.shape[0], self.coin_flip_dim).to(self.device)
        self.cfn_optimizer = optim.Adam(self.cfn.parameters(), lr=self.cfn_cfg.cfn_lr)

        self.cfn_buffer = CFNReplayBufferWrapper(
            size=cfn_cfg.cfn_replay_buffer_size,
            obs_shape=env.observation_space.shape[0],
            coin_flip_dim=self.coin_flip_dim,
            alpha=0.5
        )

        self.is_goal_env = hasattr(env, 'is_goal_env') and env.is_goal_env

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
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)

        while current_timestep < total_timesteps:

            with torch.no_grad():
                action, _ = self.actor(obs)
                action = action.cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)

            next_obs_np, reward, terminated, truncated, info = self.env.step(action)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)
            done = terminated or truncated

            intrinsic_reward = compute_intrinsic_reward(
                self.coin_flip_dim,
                self.cfn.compute_squared_output_norm(obs)
            )

            avg_rewards.append(intrinsic_reward.item() + reward)

            if self.use_cfn_prior:
                if current_timestep < 30_000 or current_timestep > 970_000:
                    with torch.no_grad():
                        obs = obs.to(self.cfn.device)
                        if obs.ndim == 1:
                            obs = obs.unsqueeze(0)

                        combined_out = self.cfn(obs, update_prior_stats=False)
                        prior_out = self.cfn.prior(obs)
                        output_norm = combined_out.norm(p=2, dim=1)
                        prior_output_norm = prior_out.norm(p=2, dim=1)
                        pseudocount_estimate = self.cfn.coin_flip_dim / (output_norm ** 2)

                        wandb.log({
                            "pseudocounts-intr": 1 / (intrinsic_reward**2 + 1e-8).item(),
                            "prior_output_norm": prior_output_norm.cpu().numpy(),
                            "output_norm": output_norm.cpu().numpy(),
                            "pseudocount_estimate": pseudocount_estimate.cpu().numpy(),
                        }, step=current_timestep)

                self.cfn(obs, update_prior_stats=True)

            if current_timestep % 1000 == 0:
                wandb.log({
                    "ext_reward": reward,
                    "int_reward": intrinsic_reward.item(),
                    "averaged_rewards": np.average(avg_rewards),
                    # "pseudocounts-intr": 1/torch.sqrt(intrinsic_reward+1e-8).item()
                }, step=current_timestep)

                avg_rewards.clear()

            self.buffer.add(
                obs=obs.cpu().numpy(),
                act=np.array(action, dtype=np.float32),
                ext_rew=np.array(reward, dtype=np.float32),
                int_rew=intrinsic_reward.cpu().numpy().astype(np.float32),
                next_obs=next_obs.cpu().numpy(),
                done=np.array(done, dtype=np.float32)
            )

            coin_flip = get_coin_flips(self.coin_flip_dim)
            self.cfn_buffer.add(
                obs=obs.cpu().numpy(),
                coin_flip=coin_flip.detach().cpu().numpy(),
                priority=1.0
            )

            if current_timestep >= learning_starts:
                sample = self.buffer.sample(self.batch_size)

                obs_batch = torch.tensor(sample["obs"], dtype=torch.float32, device=self.device)
                next_obs_batch = torch.tensor(sample["next_obs"], dtype=torch.float32, device=self.device)
                act_batch = torch.tensor(sample["act"], dtype=torch.float32, device=self.device)
                ext_rew_batch = torch.tensor(sample["ext_rew"], dtype=torch.float32, device=self.device).squeeze(-1)
                int_rew_batch = torch.tensor(sample["int_rew"], dtype=torch.float32, device=self.device).squeeze(-1)
                tm_batch = torch.tensor(sample["done"], dtype=torch.float32, device=self.device).squeeze(-1)

                # total_rew_batch = int_rew_batch + ext_rew_batch

                total_rew_batch = torch.clamp(int_rew_batch + ext_rew_batch, min=-1.0, max=1.0)

                self.update(obs_batch, act_batch, total_rew_batch, next_obs_batch, tm_batch, current_timestep)

            obs_batch_bc, coin_flip_batch_bc, indices = self.cfn_buffer.sample_with_indices(
                batch_size=self.cfn_cfg.cfn_batch_size
            )

            self.update_cfn(obs_batch_bc, coin_flip_batch_bc)

            if self.use_cfn_priority:
                self.cfn_buffer.update_priorities(
                    indices, obs_batch_bc, self.cfn, self.coin_flip_dim
                )

            obs = next_obs
            episode_return += reward + intrinsic_reward
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
                obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
                episode_return = 0
                episode_step = 0
                episode_num += 1

            if current_timestep % 10000 == 0:
                validate(self.actor, current_timestep)

                eval_envstep, eval_mean, eval_std = evaluate(self.actor, self.eval_env, current_timestep,
                                                             max_episode_steps)
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

        # save_rollout_gif(self.actor, self.env, Path(HydraConfig.get().runtime.output_dir) / "validate" / f"eval_gif{current_timestep}.gif")

        cfn_early_vs_late_training_comparison(self.cfn,
                                              eval_dir=Path(HydraConfig.get().runtime.output_dir) / "evaluate")

        evaluate_cfn_bonus_generalization(self.cfn, self.env, self.buffer)

    def update_cfn(self, obs_batch: torch.Tensor, coin_flip_batch: torch.Tensor):
        predicted_coin_flips = self.cfn(obs_batch)
        cfn_loss = F.mse_loss(predicted_coin_flips, coin_flip_batch)
        self.cfn_optimizer.zero_grad()
        cfn_loss.backward()
        self.cfn_optimizer.step()
