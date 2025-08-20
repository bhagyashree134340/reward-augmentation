import sys
import os

from matplotlib import pyplot as plt

from utils.evaluate import evaluate_dqn
from utils.validate import validate_dqn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
from pathlib import Path
from collections import deque
from hydra.core.hydra_config import HydraConfig
import random
import gymnasium as gym
import customised_doorkey

from networks.ddqn import DDQN
from utils.stats import EpisodeStats
import logging
from cpprb import ReplayBuffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


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
    def __init__(self, obs_shape, hidden_size=512):
        super(RNDModel, self).__init__()
        
        c, h, w = obs_shape
        
        self.target = nn.Sequential(
            self._layer_init(nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
            self._layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
            self._layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        
        conv_output_size = 64 * 2 * 2
        
        self.target.add_module('fc', self._layer_init(nn.Linear(conv_output_size, hidden_size)))
        
        self.predictor = nn.Sequential(
            self._layer_init(nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
            self._layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
            self._layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            self._layer_init(nn.Linear(conv_output_size, hidden_size)),
            nn.ReLU(),
            self._layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            self._layer_init(nn.Linear(hidden_size, hidden_size))
        )
        
        for param in self.target.parameters():
            param.requires_grad = False

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, gain=std)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, obs):
        # Expect obs to already be normalized to [0, 1] range
        target_output = self.target(obs)
        predictor_output = self.predictor(obs)
        return predictor_output, target_output


class DQN_RNDAgent:
    def __init__(self, env, eval_env, dqn_cfg, rnd_cfg, env_name):
        self.env = env
        self.eval_env = eval_env
        self.env_name = env_name  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_shape = env.observation_space.shape  
        act_dim = env.action_space.n

        h = self.env.unwrapped.grid.height
        w = self.env.unwrapped.grid.width
        self.visit_counts = np.zeros((h - 2, w - 2), dtype=np.int32)

        print("Original obs shape:", env.observation_space.shape)
        
        if len(self.obs_shape) == 3:
            self.obs_shape_torch = (self.obs_shape[2], self.obs_shape[0], self.obs_shape[1])  
        else:
            raise ValueError(f"Unexpected observation shape: {self.obs_shape}")
            
        print("PyTorch obs shape:", self.obs_shape_torch)

        self.q_net = DDQN(self.obs_shape_torch, act_dim, dqn_cfg.hidden_size, is_cnn=True).to(self.device)
        self.target_q_net = DDQN(self.obs_shape_torch, act_dim, dqn_cfg.hidden_size, is_cnn=True).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=dqn_cfg.lr)
        self.gamma = dqn_cfg.gamma
        self.batch_size = dqn_cfg.batch_size
        self.target_update_freq = dqn_cfg.target_update_freq

        self.replay_buffer = ReplayBuffer(
            dqn_cfg.replay_buffer_size,
            env_dict={
                "obs":      {"shape": self.obs_shape_torch, "dtype": np.uint8},
                "act":      {"shape": 1,                    "dtype": np.int16},
                "ext_rew":  {"shape": 1,                    "dtype": np.float32},
                "int_rew":  {"shape": 1,                    "dtype": np.float32}, 
                "done":     {"shape": 1,                    "dtype": np.bool_},
                "next_obs": {"shape": self.obs_shape_torch, "dtype": np.uint8},
            },
        )

        self.rnd_cfg = rnd_cfg
        self.intrinsic_coef = rnd_cfg.intrinsic_coef
        self.extrinsic_coef = rnd_cfg.extrinsic_coef

        self.rnd = RNDModel(self.obs_shape_torch).to(self.device)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_cfg.lr)

        # Initialize RMS for normalized observations (already in [0, 1] range)
        self.obs_rms = RunningMeanStd(shape=self.obs_shape_torch)
        self.reward_rms = RunningMeanStd()
        self.reward_filter = RewardForwardFilter(gamma=0.99)

    def increment_visit_counts(self):
        x, y = map(int, self.env.unwrapped.agent_pos)
        self.visit_counts[(y - 1), (x - 1)] += 1
    
    def process_obs(self, obs):
        if isinstance(obs, dict) and 'image' in obs:
            obs = obs['image']
        
        obs = np.array(obs, dtype=np.uint8)
        
        if len(obs.shape) == 3:
            obs = np.transpose(obs, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")
            
        return obs
    
    def obs_to_float_tensor(self, obs):
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs.to(dtype=torch.float32, device=self.device)
        
        if obs_tensor.max() > 1.0:
            obs_tensor = obs_tensor / 255.0
            
        return obs_tensor
    
    def update_obs_rms(self, obs):
        """
        Update observation RMS with normalized observation.
        obs should be uint8 numpy array in CHW format.
        """
        # Convert to float and normalize
        obs_float = obs.astype(np.float32) / 255.0
        # Update RMS with batch dimension
        self.obs_rms.update(obs_float[None, ...])
    
    def normalize_obs_with_rms(self, obs_tensor):

        obs_mean = torch.from_numpy(self.obs_rms.mean).float().to(self.device)
        obs_std = torch.sqrt(torch.from_numpy(self.obs_rms.var).float().to(self.device) + 1e-8)
        return (obs_tensor - obs_mean) / obs_std

    def act(self, obs, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        
        obs_tensor = self.obs_to_float_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return q_values.argmax().item()

    def compute_intrinsic_reward(self, obs_tensor):
        """
        Compute intrinsic reward for batch of observations.
        obs_tensor should be float32 tensor in [0, 1] range.
        """
        with torch.no_grad():
            norm_obs = self.normalize_obs_with_rms(obs_tensor)
            pred, target = self.rnd(norm_obs)
            intrinsic_reward = 0.5 * ((pred - target) ** 2).sum(dim=1)
            return intrinsic_reward

    def _fit_obs_rms_warmup(self):
        """Warmup phase to initialize observation RMS statistics."""
        print("Starting observation RMS warmup...")
        obs_raw, _ = self.env.reset()
        obs = self.process_obs(obs_raw)
        self.update_obs_rms(obs)
        
        for i in range(5000):
            a = self.env.action_space.sample()
            next_obs_raw, _, terminated, truncated, _ = self.env.step(a)
            obs = self.process_obs(next_obs_raw)
            self.update_obs_rms(obs)

            if terminated or truncated:
                obs_raw, _ = self.env.reset()
                obs = self.process_obs(obs_raw)
                self.update_obs_rms(obs)
        
        print("Observation RMS warmup completed.")
        print(f"Obs mean: {self.obs_rms.mean.mean():.4f}, Obs std: {np.sqrt(self.obs_rms.var.mean()):.4f}")

    def train(self, total_timesteps, max_episode_steps):
        current_timestep = 0
        episode_return = 0
        episode_step = 0
        episode_num = 0
        
        # Initialize environment and RMS
        obs_raw, _ = self.env.reset()
        obs = self.process_obs(obs_raw)
        self.update_obs_rms(obs)
        # self.increment_visit_counts()
        
        avg_rewards = []
        avg_int_rewards = []
        avg_ext_rewards = []
        stats = EpisodeStats([], [], [])
        epsilon = self.rnd_cfg.epsilon_start

        # Warmup observation RMS
        self._fit_obs_rms_warmup()

        obs_raw, _ = self.env.reset()
        obs = self.process_obs(obs_raw)
        self.update_obs_rms(obs)
        self.increment_visit_counts()

        while current_timestep < total_timesteps:
            action = self.act(obs, epsilon)

            next_obs_raw, ext_reward, terminated, truncated, _ = self.env.step(action)
            next_obs = self.process_obs(next_obs_raw)
            self.increment_visit_counts()

            done = bool(truncated) or bool(terminated)
           
            if ext_reward > 0:
                wandb.log({
                    "ext_rew": ext_reward
                }, step=current_timestep)

            next_obs_tensor = self.obs_to_float_tensor(next_obs).unsqueeze(0)
        
            int_reward_tensor = self.compute_intrinsic_reward(next_obs_tensor)
            int_reward = int_reward_tensor.item()

            discounted_r = self.reward_filter.update(int_reward)
            self.reward_rms.update(np.array([discounted_r]))
            norm_int_reward = int_reward / np.sqrt(np.maximum(self.reward_rms.var, 1e-8))

            if current_timestep % 1000 == 0:
                agent_x, agent_y = map(int, self.env.unwrapped.agent_pos)
                wandb.log(
                    {f"int_reward/cell({agent_y - 1},{agent_x - 1})": norm_int_reward},
                    step=current_timestep
                )

            total_reward = self.extrinsic_coef * ext_reward + self.intrinsic_coef * norm_int_reward
            
            avg_rewards.append(total_reward)
            avg_int_rewards.append(norm_int_reward)
            avg_ext_rewards.append(ext_reward)

            # Store in replay buffer
            self.replay_buffer.add(
                obs=obs,
                act=action,
                ext_rew=ext_reward,
                int_rew=norm_int_reward,   
                done=done,
                next_obs=next_obs,
            )

            # Update RND network
            mask_prob = getattr(self.rnd_cfg, 'rnd_mask_prob', 0.25)
            if current_timestep > self.rnd_cfg.learning_starts:
                batch_obs = self.replay_buffer.sample(self.batch_size)["obs"]
                obs_tensor = self.obs_to_float_tensor(batch_obs)
                norm_obs = self.normalize_obs_with_rms(obs_tensor)

                pred, tgt = self.rnd(norm_obs)

                mask = torch.rand(self.batch_size, device=self.device) < mask_prob
                if mask.any():
                    forward_loss = F.mse_loss(pred[mask], tgt.detach()[mask])
                    self.rnd_optimizer.zero_grad()
                    forward_loss.backward()
                    self.rnd_optimizer.step()

            # Update DQN
            if current_timestep > self.rnd_cfg.learning_starts and self.replay_buffer.get_stored_size() >= self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                self.update_dqn(batch)

            # Logging
            if current_timestep % 1000 == 0 and current_timestep > 0:
                wandb.log({
                    "a/ext_reward": ext_reward,
                    "a/int_reward": int_reward,
                    "a/epsilon": epsilon,
                }, step=current_timestep)
                avg_rewards.clear()
                avg_int_rewards.clear()
                avg_ext_rewards.clear()

            # Update observation RMS and move to next step
            self.update_obs_rms(next_obs)
            obs = next_obs
            episode_return += total_reward
            episode_step += 1
            current_timestep += 1

            # Update target network
            if current_timestep % self.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            # Handle episode end
            if done or episode_step >= max_episode_steps:
                stats.episode_rewards.append(episode_return)
                stats.episode_lengths.append(episode_step)
                stats.timesteps_on_ep_end.append(current_timestep)

                wandb.log({
                    "charts/episodic_return": episode_return,
                    "charts/episodic_length": episode_step,
                    "charts/episode_num": episode_num
                }, step=current_timestep)

                log.info(
                    f"Episode {episode_num} | Steps: {episode_step} | "
                    f"Return: {episode_return:.2f} | Ext Reward: {ext_reward} | Total Timesteps: {current_timestep}"
                )

                if epsilon > self.rnd_cfg.epsilon_end:
                    epsilon *= self.rnd_cfg.epsilon_decay

                # Reset environment
                obs_raw, _ = self.env.reset()
                obs = self.process_obs(obs_raw)
                self.update_obs_rms(obs)
                self.increment_visit_counts()
                episode_return = 0
                episode_step = 0
                episode_num += 1

            # Periodic evaluation and plotting
            if current_timestep % 10000 == 0 and current_timestep > 0:
                validate_dqn(self, current_timestep, save_dir="checkpoints_rnd")
                evaluate_dqn(self, self.eval_env, current_timestep, save_dir="eval_rnd")

            if current_timestep % 1000 == 0:
                self.plot_intrinsic_vs_true_bonus_heatmap_minigrid_rnd(current_timestep)

    def plot_intrinsic_vs_true_bonus_heatmap_minigrid_rnd(self, step):
        """
        Plots and logs heatmaps comparing true bonus (from visitation count)
        vs intrinsic reward bonus from RND. Uses base env to ensure correct rendering.
        """
        visit_counts = self.visit_counts
        h, w = visit_counts.shape

        # Compute true bonus
        true_bonus = 1.0 / np.sqrt(visit_counts + 1e-8)
        intrinsic_bonus = np.full_like(true_bonus, fill_value=np.nan, dtype=np.float32)

        base_env = self.eval_env.unwrapped  # Access base MiniGrid env

        for y in range(h):
            for x in range(w):
                try:
                    base_env.reset()
                    base_env.agent_pos = (x + 1, y + 1)  # Offset for wall
                    base_env.agent_dir = np.random.randint(0, 4)  # Random direction
                    base_env.step(base_env.actions.toggle)  # Dummy step to refresh visuals

                    obs_raw = base_env.render()  # Must call render on base env
                    obs = self.process_obs(obs_raw)
                    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                    if obs_tensor.max() > 1.0:
                        obs_tensor = obs_tensor / 255.0

                    # Resize if needed
                    c, h_rms, w_rms = self.obs_rms.mean.shape
                    if obs_tensor.shape[-2:] != (h_rms, w_rms):
                        obs_tensor = F.interpolate(obs_tensor, size=(h_rms, w_rms), mode='bilinear', align_corners=False)

                    with torch.no_grad():
                        bonus = self.compute_intrinsic_reward(obs_tensor).item()
                        norm_int_reward = bonus / np.sqrt(np.maximum(self.reward_rms.var, 1e-8))
                        intrinsic_bonus[y, x] = norm_int_reward

                except Exception as e:
                    print(f"Failed at ({x},{y}): {e}")
                    continue

        # Mask unvisited cells
        true_bonus_masked = np.ma.masked_where(visit_counts == 0, true_bonus)
        intrinsic_bonus_masked = np.ma.masked_where(visit_counts == 0, intrinsic_bonus)

        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        true_vmin = np.nanmin(true_bonus_masked)
        true_vmax = np.nanmax(true_bonus_masked)
        # Percentile clipping to avoid extreme outliers
        clipped_intrinsic_bonus = intrinsic_bonus.copy()
        vmin_clip = np.nanpercentile(clipped_intrinsic_bonus, 1)
        vmax_clip = np.nanpercentile(clipped_intrinsic_bonus, 99)

        # Clip values for visualization only
        clipped_intrinsic_bonus = np.clip(clipped_intrinsic_bonus, vmin_clip, vmax_clip)
        rnd_vmin, rnd_vmax = vmin_clip, vmax_clip

        # True Bonus Map
        im0 = axs[0].imshow(true_bonus_masked, cmap="magma", vmin=true_vmin, vmax=true_vmax)
        axs[0].set_title("True Bonus (1/sqrt(count))")
        for y in range(h):
            for x in range(w):
                if visit_counts[y, x] > 0:
                    axs[0].text(x, y, f"{visit_counts[y, x]}", ha='center', va='center',
                                color='white' if true_bonus[y, x] < (true_vmin + true_vmax) / 2 else 'black')
        fig.colorbar(im0, ax=axs[0])

        # RND Intrinsic Bonus Map
        im1 = axs[1].imshow(clipped_intrinsic_bonus, cmap="viridis", vmin=rnd_vmin, vmax=rnd_vmax)
        axs[1].set_title("RND Intrinsic Bonus")
        fig.colorbar(im1, ax=axs[1])

        for ax in axs:
            ax.set_xticks(range(w))
            ax.set_yticks(range(h))

        plt.tight_layout()
        wandb.log({f"rnd/true_vs_intrinsic_bonus_heatmap": wandb.Image(fig)}, step=step)
        plt.close(fig)

    def update_dqn(self, batch):
        obs = self.obs_to_float_tensor(batch["obs"])
        next_obs = self.obs_to_float_tensor(batch["next_obs"])

        act = torch.from_numpy(batch["act"].squeeze(-1)).long().to(self.device)
        ext_rew = torch.from_numpy(batch["ext_rew"].squeeze(-1)).float().to(self.device)
        int_rew = torch.from_numpy(batch["int_rew"].squeeze(-1)).float().to(self.device)
        dones = torch.from_numpy(batch["done"].squeeze(-1)).float().to(self.device)

        total_rew = self.extrinsic_coef * ext_rew + self.intrinsic_coef * int_rew

        q_vals = self.q_net(obs)
        q_val = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_vals = self.q_net(next_obs)
            next_actions = next_q_vals.argmax(1)
            target_q_vals = self.target_q_net(next_obs)
            max_next_q_vals = target_q_vals.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = total_rew + (1 - dones) * self.gamma * max_next_q_vals

        loss = F.mse_loss(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def print_visit_counts(self):
        print("\nTrue visit counts:")
        for row in self.visit_counts:
            print(" ".join(f"{v:4d}" for v in row))


from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper


def main():
    wandb.init(project="dqn", name="rnd")
    ENV_NAME = "Fixed-DoorKey-6x6-v0"  

    env = gym.make(
        "Fixed-DoorKey-v0",
        size=10,
        key_pos=(1, 8),           # bottom-left room
        door_pos=(5, 5),          # middle vertical wall
        goal_pos=(8, 5),          # right room
        agent_start_pos=(1, 1),   # top-left
        agent_start_dir=0,
        disable_env_checker=True,
        max_episode_steps=400,
        render_mode="rgb_array"  # Use RGB rendering for evaluation
    )
    
    env = customised_doorkey.NoDropWrapper(env)
    env = FullyObsWrapper(env)
    env = RGBImgObsWrapper(env, tile_size=4)
    env = ImgObsWrapper(env)

    eval_env = gym.make(
        "Fixed-DoorKey-v0",
        size=10,
        key_pos=(1, 8),           # bottom-left room
        door_pos=(5, 5),          # middle vertical wall
        goal_pos=(8, 5),          # right room
        agent_start_pos=(1, 1),   # top-left
        agent_start_dir=0,
        disable_env_checker=True,
        max_episode_steps=400,
        render_mode="rgb_array"  # Use RGB rendering for evaluation
    )
    eval_env = customised_doorkey.NoDropWrapper(eval_env)
    eval_env = FullyObsWrapper(eval_env)
    eval_env = RGBImgObsWrapper(eval_env, tile_size=4)
    eval_env = ImgObsWrapper(eval_env)

    max_episode_steps = 400
    total_timesteps = 1000_000
    

    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": 128,
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "replay_buffer_size": 1_000_000,
        "target_update_freq": 2000
    })

    rnd_cfg = type("RNDConfig", (), {
        "intrinsic_coef": 1.0,
        "extrinsic_coef": 2.0,
        "lr": 1e-3,
        "learning_starts": 1000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.9998,
        "rnd_mask_prob": 0.25
    })

    agent = DQN_RNDAgent(
        env=env,
        env_name = ENV_NAME,
        eval_env=eval_env,
        dqn_cfg=dqn_cfg,
        rnd_cfg=rnd_cfg
    )

    start = time.time()
    agent.train(total_timesteps=total_timesteps, max_episode_steps=max_episode_steps)
    end = time.time()
    print(f"Training finished in {(end - start) / 60:.2f} minutes.")

    agent.print_visit_counts()

if __name__ == "__main__":
    main()