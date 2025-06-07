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
from utils.plots import plot_and_save_training_metrics, plot_eval_curve, cfn_early_vs_late_training_comparison
from utils.stats import EpisodeStats
import logging

from utils.validate import validate

log = logging.getLogger(__name__)


class SACCFNAgent(SACAgent):
    def __init__(self, env, eval_env, cfn_cfg, **kwargs):
        super().__init__(env, eval_env, **kwargs)

        self.cfn_cfg = cfn_cfg
        self.use_cfn_prior = cfn_cfg.use_cfn_prior
        self.use_cfn_priority = cfn_cfg.use_cfn_priority
        self.coin_flip_dim = cfn_cfg.cfn_coin_flip_dim

        self.cfn = CoinFlipNetwork(env.observation_space.shape[0], self.coin_flip_dim)
        self.cfn_optimizer = optim.Adam(self.cfn.parameters(), lr=self.cfn_cfg.cfn_lr)

        self.cfn_buffer = CFNReplayBufferWrapper(
            size=cfn_cfg.cfn_replay_buffer_size,
            obs_shape=env.observation_space.shape[0],
            coin_flip_dim=self.coin_flip_dim,
            alpha=0.5
        )

        self.is_goal_env = hasattr(env, 'is_goal_env') and env.is_goal_env

        # if self.is_goal_env:
        #     # Get dimensions from the original environment
        #     obs_space = env.env.observation_space['observation']
        #     goal_space = env.env.observation_space['desired_goal']
        #     action_space = env.action_space
        #
        #     # Replace regular buffer with HER buffer
        #     self.buffer = HindsightReplayBuffer(
        #         size=16384,
        #         max_episode_len=500000,
        #         env_dict={
        #             "obs": {"shape": obs_space.shape},
        #             "act": {"shape": action_space.shape},
        #             "ext_rew": {},
        #             "int_rew": {},
        #             "next_obs": {"shape": obs_space.shape},
        #             "done": {},
        #             "achieved_goal": {"shape": goal_space.shape},
        #             "desired_goal": {"shape": goal_space.shape}
        #         },
        #         goal_func=self._goal_func,
        #         reward_func=self._reward_func,
        #         her_strategy="future",
        #         her_ratio=0.8
        #     )
        #     log.info("Using HER buffer for goal-conditioned environment")
        # else:
        #     log.warning("Environment is not goal-conditioned, using regular buffer")

    # def _goal_func(self, episode):
    #     """Extract achieved goals from episode for HER"""
    #     return episode["achieved_goal"]
    #
    # def _reward_func(self, achieved_goal, desired_goal, info):
    #     """Compute reward for HER relabeling"""
    #     # Use the environment's reward function
    #     return self.env.env.compute_reward(achieved_goal, desired_goal, info)
    #
    # def _get_obs_dict(self):
    #     """Get observation dictionary from the unwrapped environment"""
    #     env = self.env
    #     # Unwrap until we find the actual Fetch environment
    #     while hasattr(env, 'env'):
    #         env = env.env
    #     return env._get_obs()

    def train(self, total_timesteps, max_steps, learning_starts=5000):
        current_timestep = 0
        episode_return = 0
        episode_step = 0
        episode_num = 0
        episode_lengths = []
        episode_rewards = []
        timesteps_on_ep_end = []

        obs, _ = self.env.reset()
        # obs_dict, _ = self.env.env.reset()

        while current_timestep < total_timesteps:
            with torch.no_grad():
                action, _ = self.actor(torch.as_tensor(obs).float())
                action = action.cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)

            next_obs, reward, terminated, truncated, info = self.env.step(action)

            done = terminated or truncated

            intrinsic_reward = compute_intrinsic_reward(
                self.coin_flip_dim,
                self.cfn.compute_squared_output_norm(torch.as_tensor(obs).float())
            )

            if self.use_cfn_prior:
                self.cfn(torch.as_tensor(obs).float(), update_prior_stats=True)

            if current_timestep%1000 == 0:
                wandb.log({
                    "ext_reward": reward,
                    "int_reward": intrinsic_reward,
                    # "int rew / total rew": intrinsic_reward/(reward+intrinsic_reward)
                })

            # if self.is_goal_env:
            #     # Get the current observation dict from the unwrapped environment
            #     current_obs_dict = self._get_obs_dict()
            #
            #     # Store with goal information for HER
            #     self.buffer.add(
            #         obs=current_obs_dict['observation'].astype(np.float32),
            #         act=np.array(action, dtype=np.float32),
            #         ext_rew=np.array(reward, dtype=np.float32),
            #         int_rew=np.array(intrinsic_reward, dtype=np.float32),
            #         next_obs=current_obs_dict['observation'].astype(np.float32),  # Will be updated next step
            #         done=np.array(done, dtype=np.float32),
            #         achieved_goal=current_obs_dict['achieved_goal'].astype(np.float32),
            #         desired_goal=current_obs_dict['desired_goal'].astype(np.float32)
            #     )
            # else:
            # Regular buffer storage for non-goal environments
            self.buffer.add(
                obs=np.array(obs, dtype=np.float32),
                act=np.array(action, dtype=np.float32),
                ext_rew=np.array(reward, dtype=np.float32),
                int_rew=np.array(intrinsic_reward, dtype=np.float32),
                next_obs=np.array(next_obs, dtype=np.float32),
                done=np.array(done, dtype=np.float32)
            )

            coin_flip = get_coin_flips(self.coin_flip_dim)
            # self.cfn_buffer.store(torch.tensor(obs), coin_flip.detach())

            self.cfn_buffer.add(
                obs=np.array(obs, dtype=np.float32),
                coin_flip=coin_flip.detach().numpy(),
                priority=1.0
            )

            if current_timestep >= learning_starts:

                # if self.buffer.get_stored_size() >= self.batch_size:
                sample = self.buffer.sample(self.batch_size)

                # if self.is_goal_env:
                #     # For goal environments, concatenate obs with desired_goal for policy input
                #     obs_batch = torch.tensor(
                #         np.concatenate([sample["obs"], sample["desired_goal"]], axis=1),
                #         dtype=torch.float32
                #     )
                #     next_obs_batch = torch.tensor(
                #         np.concatenate([sample["next_obs"], sample["desired_goal"]], axis=1),
                #         dtype=torch.float32
                #     )
                # else:
                obs_batch = torch.tensor(sample["obs"], dtype=torch.float32)
                next_obs_batch = torch.tensor(sample["next_obs"], dtype=torch.float32)

                act_batch = torch.tensor(sample["act"], dtype=torch.float32)
                ext_rew_batch = torch.tensor(sample["ext_rew"], dtype=torch.float32).squeeze(-1)
                int_rew_batch = torch.tensor(sample["int_rew"], dtype=torch.float32).squeeze(-1)
                tm_batch = torch.tensor(sample["done"], dtype=torch.float32).squeeze(-1)

                total_rew_batch = ext_rew_batch + int_rew_batch

                self.update(obs_batch, act_batch, total_rew_batch, next_obs_batch, tm_batch, current_timestep)

                # if self.cfn_buffer.get_stored_size() >= self.cfn_cfg.cfn_batch_size:
                # Sampling + updating
                obs_batch_bc, coin_flip_batch_bc, _ = self.cfn_buffer.sample_and_update_priorities(
                    batch_size=self.cfn_cfg.cfn_batch_size,
                    cfn=self.cfn,
                    coin_flip_dim=self.coin_flip_dim,
                    use_cfn_priority=self.use_cfn_priority
                )

                self.update_cfn(obs_batch_bc, coin_flip_batch_bc)

            obs = next_obs
            episode_return = episode_return + reward + intrinsic_reward
            episode_step += 1
            current_timestep += 1

            if done or episode_step >= max_steps:
                episode_lengths.append(episode_step)
                episode_rewards.append(episode_return)
                timesteps_on_ep_end.append(current_timestep)

                wandb.log({
                    "episode return": episode_return,
                    "episode length": episode_step,
                    "episode num": episode_num
                    # "intrinsic rewards": intrinsic_reward,
                    # "reward": reward
                }, step=current_timestep)

                log.info(
                    f"Episode {episode_num} | Steps: {episode_step} | "
                    f"Return: {episode_return:.2f} | Total Timesteps: {current_timestep}"
                )

                obs, _ = self.env.reset()
                episode_return = 0
                episode_step = 0
                episode_num += 1

            # Save actor and evaluate periodically
            if current_timestep % 1000 == 0:
                validate(self.actor, current_timestep)

                eval_envstep, eval_mean, eval_std = evaluate(self.actor, self.eval_env, current_timestep, max_steps)
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

        # Save plot
        plot_path = Path(
            HydraConfig.get().runtime.output_dir) / "validate" / f"eval_plot_step{current_timestep}.png"
        plot_eval_curve(self.eval_envsteps, self.eval_means, self.eval_stds, plot_path)

        save_rollout_gif(self.actor, self.env,  Path(
            HydraConfig.get().runtime.output_dir) / "validate" / f"eval_gif{current_timestep}.gif")

        cfn_early_vs_late_training_comparison(self.cfn,
                                              eval_dir=Path(HydraConfig.get().runtime.output_dir) / "evaluate")

        evaluate_cfn_bonus_generalization(self.cfn, self.env, self.buffer)

    def update_cfn(self, obs_batch: torch.Tensor, coin_flip_batch: torch.Tensor):
        """
        Update the Coin Flip Network with the current batch of states.

        :param coin_flip_batch:
        :param obs_batch: Batch of current observations.
        """
        # Compute the predicted coin flip vectors for the batch of states
        predicted_coin_flips = self.cfn(obs_batch)

        # Compute the loss with respect to the coin flip vectors
        # (Assuming coin flip vectors are stored in the CFN replay buffer)

        # i think this will be from the buffer and the get_coin_flips_for_batch will be stored into the cfn buffer
        # coin_flips_batch = self.get_coin_flips_for_batch(obs_batch)

        cfn_loss = F.mse_loss(predicted_coin_flips, coin_flip_batch)

        # Backpropagate and update the CFN
        self.cfn_optimizer.zero_grad()
        cfn_loss.backward()
        self.cfn_optimizer.step()
