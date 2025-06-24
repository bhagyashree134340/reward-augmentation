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
from utils.plots import log_cfn_stats_to_wandb, plot_and_save_training_metrics, plot_eval_curve, cfn_early_vs_late_training_comparison
from utils.stats import EpisodeStats
import logging
from gym.wrappers.normalize import RunningMeanStd

from utils.validate import validate

log = logging.getLogger(__name__)

from torch import nn

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
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

        self.target = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            target = self.target(x)
        pred = self.predictor(x)
        return pred, target


class SACRNDAgent(SACAgent):
    def __init__(self, env, rnd_cfg, **kwargs):
        super().__init__(env, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_rnd = True #TODO: change this to take from config
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
        avg_rewards=[]

        obs_np, _ = self.env.reset()
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)


        while current_timestep < total_timesteps:

            gripper_pos = obs[:3]
            goal_pos = obs[-3:]
            distance = np.linalg.norm(gripper_pos.cpu().numpy() - goal_pos.cpu().numpy())
            wandb.log({"gripper-goal-dist":distance}, step=current_timestep)

            with torch.no_grad():
                action, _ = self.actor(obs)
                action = action.cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)

            next_obs_np, reward, terminated, truncated, info = self.env.step(action)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)
            done = terminated or truncated


            obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            pred, target = self.rnd(obs_tensor)
            int_rew = F.mse_loss(pred, target.detach(), reduction='none').mean().item()
            normed_int_rew = int_rew / np.sqrt(self.reward_rms.var + 1e-8)
            total_rew = reward + self.beta * normed_int_rew

            self.buffer.add(
                obs=np.array(obs, dtype=np.float32),
                act=np.array(action, dtype=np.float32),
                ext_rew=np.array(reward, dtype=np.float32),
                int_rew=np.array(normed_int_rew, dtype=np.float32),
                next_obs=np.array(next_obs, dtype=np.float32),
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

            # if current_timestep >= 30_000:

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

                eval_envstep, eval_mean, eval_std = evaluate(self.actor, self.eval_env, current_timestep, max_episode_steps)
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