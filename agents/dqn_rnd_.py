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
from gymnasium.wrappers.utils import RunningMeanStd
import customised_doorkey

from networks.ddqn import DDQN
from utils.stats import EpisodeStats
import logging
from cpprb import ReplayBuffer
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
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
    def __init__(self, obs_shape, hidden_size=128):
        super(RNDModel, self).__init__()
        c, h, w = obs_shape
        
        # Simpler encoder for small 10x10 grids
        def make_encoder():
            return nn.Sequential(
                nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )

        self.target_encoder = make_encoder()
        self.predictor_encoder = make_encoder()
        
        # Calculate feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            feature_dim = self.target_encoder(dummy_input).shape[1]
        
        # Smaller heads
        self.target_head = nn.Linear(feature_dim, hidden_size)
        self.predictor_head = nn.Linear(feature_dim, hidden_size)

        # Freeze target network
        for p in list(self.target_encoder.parameters()) + list(self.target_head.parameters()):
            p.requires_grad = False

    def forward(self, obs):
        target_features = self.target_encoder(obs)
        target_output = self.target_head(target_features)
        
        predictor_features = self.predictor_encoder(obs)
        predictor_output = self.predictor_head(predictor_features)
        
        return predictor_output, target_output

class DQN_RNDAgent:
    def __init__(self, env, eval_env, dqn_cfg, rnd_cfg, env_name):
        self.env = env
        self.eval_env = eval_env
        self.env_name = env_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_shape = env.observation_space.shape
        act_dim = env.action_space.n

        # visit counts for simple "true bonus"
        h = self.env.unwrapped.grid.height
        w = self.env.unwrapped.grid.width
        self.visit_counts = np.zeros((h - 2, w - 2), dtype=np.int32)

        print("Original obs shape:", env.observation_space.shape)
        if len(self.obs_shape) == 3:
            self.obs_shape_torch = (self.obs_shape[2], self.obs_shape[0], self.obs_shape[1])
        else:
            raise ValueError(f"Unexpected observation shape: {self.obs_shape}")
        print("PyTorch obs shape:", self.obs_shape_torch)

        # q nets
        self.q_net = DDQN(self.obs_shape_torch, act_dim, dqn_cfg.hidden_size, is_cnn=True).to(self.device)
        self.target_q_net = DDQN(self.obs_shape_torch, act_dim, dqn_cfg.hidden_size, is_cnn=True).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=dqn_cfg.lr)
        self.gamma = dqn_cfg.gamma
        self.batch_size = dqn_cfg.batch_size
        self.target_update_freq = dqn_cfg.target_update_freq
        self.train_every = dqn_cfg.train_every
        self._update_steps = 0
        self._step_count = 0

        # replay
        self.replay_buffer = ReplayBuffer(
            dqn_cfg.replay_buffer_size,
            env_dict={
                "obs": {"shape": self.obs_shape_torch, "dtype": np.uint8},
                "act": {"shape": 1, "dtype": np.int16},
                "ext_rew": {"shape": 1, "dtype": np.float32},
                "int_rew": {"shape": 1, "dtype": np.float32},
                "done": {"shape": 1, "dtype": np.bool_},
                "next_obs": {"shape": self.obs_shape_torch, "dtype": np.uint8},
            },
        )

        # rnd config
        self.rnd_cfg = rnd_cfg
        self.intrinsic_coef = rnd_cfg.intrinsic_coef
        self.extrinsic_coef = rnd_cfg.extrinsic_coef

        # rnd nets + opt
        self.rnd = RNDModel(self.obs_shape_torch).to(self.device)
        self.rnd_optimizer = torch.optim.Adam(
            list(self.rnd.predictor_encoder.parameters()) + list(self.rnd.predictor_head.parameters()), 
            lr=rnd_cfg.lr
        )

        # running stats - separate for obs and rewards as per paper
        self.obs_rms = RunningMeanStd(shape=self.obs_shape_torch)
        self.reward_rms = RunningMeanStd(shape=())  # scalar shape
        self.reward_filter = RewardForwardFilter(gamma=0.99)
        
        # episode intrinsic rewards for normalization
        self.episode_intrinsic_rewards = deque(maxlen=100)

    def increment_visit_counts(self):
        x, y = map(int, self.env.unwrapped.agent_pos)
        self.visit_counts[(y - 1), (x - 1)] += 1

    def process_obs(self, obs):
        # make chw uint8
        if isinstance(obs, dict) and "image" in obs:
            obs = obs["image"]
        obs = np.array(obs, dtype=np.uint8)
        if len(obs.shape) == 3:
            obs = np.transpose(obs, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")
        return obs

    def obs_to_float_tensor(self, obs):
        # to float [0,1]
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs.to(dtype=torch.float32, device=self.device)
        if obs_tensor.max() > 1.0:
            obs_tensor = obs_tensor / 255.0
        return obs_tensor

    def act(self, obs, epsilon):
        # eps-greedy policy
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        obs_tensor = self.obs_to_float_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return q_values.argmax().item()

    def compute_intrinsic_reward(self, obs_tensor):
        # Faithful RND: whitening + clipping as per paper
        if len(obs_tensor.shape) == 3:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        obs_mean = torch.from_numpy(self.obs_rms.mean).float().to(self.device)
        obs_var = torch.from_numpy(self.obs_rms.var).float().to(self.device)
        
        # Clip variance to avoid division by zero
        obs_var = torch.clamp(obs_var, min=1e-4)
        
        normalized_obs = (obs_tensor - obs_mean) / torch.sqrt(obs_var)
        normalized_obs = torch.clamp(normalized_obs, -5.0, 5.0)
        
        with torch.no_grad():
            pred, target = self.rnd(normalized_obs)
            # MSE between predictor and target
            intrinsic_reward = F.mse_loss(pred, target, reduction='none').mean(dim=-1)
        
        return intrinsic_reward

    def _fit_obs_rms_warmup(self):
        # Collect observations for RMS initialization
        print("Starting observation RMS warmup...")
        obs_buffer = []
        
        for _ in range(500):
            obs_raw, _ = self.env.reset()
            obs = self.process_obs(obs_raw)
            obs_tensor = self.obs_to_float_tensor(obs)
            obs_buffer.append(obs_tensor.cpu().numpy())
            
            for _ in range(400):  
                a = self.env.action_space.sample()
                next_obs_raw, _, terminated, truncated, _ = self.env.step(a)
                obs = self.process_obs(next_obs_raw)
                obs_tensor = self.obs_to_float_tensor(obs)
                obs_buffer.append(obs_tensor.cpu().numpy())
                
                if terminated or truncated:
                    break
        
        # Update RMS with collected observations
        obs_array = np.stack(obs_buffer)
        self.obs_rms.update(obs_array)
        
        print("Observation RMS warmup completed.")
        print(f"Obs mean: {self.obs_rms.mean.mean():.4f}, Obs std: {np.sqrt(self.obs_rms.var.mean()):.4f}")

    def train(self, total_timesteps, max_episode_steps):
        current_timestep = 0
        episode_return = 0
        episode_ext_return = 0
        episode_int_return = 0
        episode_step = 0
        episode_num = 0
        episode_intrinsic_rewards_list = []

        stats = EpisodeStats([], [], [])
        epsilon = self.rnd_cfg.epsilon_start

        # Initialize observation RMS
        self._fit_obs_rms_warmup()

        obs_raw, _ = self.env.reset()
        obs = self.process_obs(obs_raw)
        self.increment_visit_counts()

        while current_timestep < total_timesteps:
            action = self.act(obs, epsilon)

            next_obs_raw, ext_reward, terminated, truncated, _ = self.env.step(action)
            next_obs = self.process_obs(next_obs_raw)
            self.increment_visit_counts()
            done = bool(truncated) or bool(terminated)

            if ext_reward>0:
                wandb.log({"ext_reward": ext_reward}, step=current_timestep)

            # Update observation RMS with new observation
            next_obs_tensor = self.obs_to_float_tensor(next_obs)
            self.obs_rms.update(next_obs_tensor.unsqueeze(0).cpu().numpy())

            # Compute intrinsic reward
            int_reward_tensor = self.compute_intrinsic_reward(next_obs_tensor)
            int_reward = float(int_reward_tensor.item())
            episode_intrinsic_rewards_list.append(int_reward)

            # Normalize intrinsic reward by running standard deviation as per paper
            discounted_return = self.reward_filter.update(int_reward)
            self.reward_rms.update(np.array([discounted_return]))
            
            # Avoid division by zero
            reward_std = np.sqrt(self.reward_rms.var + 1e-8)
            normalized_int_reward = int_reward / reward_std

            # Combine rewards - paper uses simple addition
            total_reward = self.extrinsic_coef * ext_reward + self.intrinsic_coef * normalized_int_reward

            # Store transition
            self.replay_buffer.add(
                obs=obs,
                act=action,
                ext_rew=ext_reward,
                int_rew=normalized_int_reward,
                done=done,
                next_obs=next_obs,
            )

            self._step_count += 1
            if (current_timestep > self.rnd_cfg.learning_starts and 
                self.replay_buffer.get_stored_size() >= self.batch_size):
                
                # Sample batch for both RND and DQN updates
                batch = self.replay_buffer.sample(self.batch_size)
                
                # RND update - use current observations as per paper
                self.update_rnd(batch)
                
                # DQN update on same batch
                self.update_dqn(batch)
                self._update_steps += 1

                # Update target network
                if self._update_steps % self.target_update_freq == 0:
                    self.target_q_net.load_state_dict(self.q_net.state_dict())

                epsilon = max(self.rnd_cfg.epsilon_end, epsilon * self.rnd_cfg.epsilon_decay)

            # Logging
            if current_timestep % 1000 == 0 and current_timestep > 0:
                wandb.log({
                    "rewards/ext_reward": ext_reward,
                    "rewards/int_reward": int_reward,
                    "rewards/norm_int_reward": normalized_int_reward,
                    "rewards/total_reward": total_reward,
                    "exploration/epsilon": epsilon,
                    "diagnostics/reward_std": reward_std,
                }, step=current_timestep)

            # Update state and episode tracking
            obs = next_obs
            episode_return += total_reward
            episode_ext_return += ext_reward
            episode_int_return += normalized_int_reward
            episode_step += 1
            current_timestep += 1

            # Episode end
            if done or episode_step >= max_episode_steps:
                # Store episode intrinsic rewards for potential normalization
                if episode_intrinsic_rewards_list:
                    self.episode_intrinsic_rewards.append(np.mean(episode_intrinsic_rewards_list))
                
                stats.episode_rewards.append(episode_return)
                stats.episode_lengths.append(episode_step)
                stats.timesteps_on_ep_end.append(current_timestep)

                wandb.log({
                    "charts/episodic_return": episode_return,
                    "charts/episodic_ext_return": episode_ext_return,
                    "charts/episodic_int_return": episode_int_return,
                    "charts/episodic_length": episode_step,
                    "charts/episode_num": episode_num,
                }, step=current_timestep)

                log.info(
                    f"Episode {episode_num} | Steps: {episode_step} | "
                    f"Total Return: {episode_return:.2f} | Ext: {episode_ext_return:.2f} | "
                    f"Int: {episode_int_return:.2f} | Timesteps: {current_timestep}"
                )

                # Reset for next episode
                obs_raw, _ = self.env.reset()
                obs = self.process_obs(obs_raw)
                self.increment_visit_counts()
                episode_return = 0
                episode_ext_return = 0
                episode_int_return = 0
                episode_step = 0
                episode_num += 1
                episode_intrinsic_rewards_list = []

            # Periodic evaluation and visualization
            if current_timestep % 50_000 == 0 and current_timestep > 0:
                validate_dqn(self, current_timestep, save_dir="checkpoints_rnd")
                evaluate_dqn(self, self.eval_env, current_timestep, save_dir="eval_rnd")
                self.log_action_gap(current_timestep)

            if current_timestep % 100_000 == 0 and current_timestep > 0:
                self.plot_intrinsic_vs_true_bonus_heatmap_minigrid_rnd(current_timestep)

    def update_rnd(self, batch):
        """Update RND predictor network"""
        # Use current observations for RND update as per paper
        obs_batch = self.obs_to_float_tensor(batch["obs"])
        
        # Normalize observations
        obs_mean = torch.from_numpy(self.obs_rms.mean).float().to(self.device)
        obs_var = torch.from_numpy(self.obs_rms.var).float().to(self.device)
        obs_var = torch.clamp(obs_var, min=1e-4)
        
        normalized_obs = (obs_batch - obs_mean) / torch.sqrt(obs_var)
        normalized_obs = torch.clamp(normalized_obs, -5.0, 5.0)

        # Forward pass
        pred, target = self.rnd(normalized_obs)
        
        # MSE loss
        rnd_loss = F.mse_loss(pred, target, reduction='none').mean(dim=-1)
        
        # Apply mask for training stability (keep only a fraction as per paper)
        mask = torch.rand_like(rnd_loss) < self.rnd_cfg.predictor_keep_ratio
        if mask.sum() == 0:
            mask = torch.ones_like(rnd_loss)
        
        loss = (rnd_loss * mask.float()).sum() / mask.float().sum()

        # Update predictor
        self.rnd_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.rnd.predictor_encoder.parameters()) + list(self.rnd.predictor_head.parameters()), 
            40.0
        )
        self.rnd_optimizer.step()

    def update_dqn(self, batch):
        """Update DQN networks"""
        obs = self.obs_to_float_tensor(batch["obs"])
        next_obs = self.obs_to_float_tensor(batch["next_obs"])

        act = torch.from_numpy(batch["act"].squeeze(-1)).long().to(self.device)
        ext = torch.from_numpy(batch["ext_rew"].squeeze(-1)).float().to(self.device)
        int_norm = torch.from_numpy(batch["int_rew"].squeeze(-1)).float().to(self.device)
        done = torch.from_numpy(batch["done"].squeeze(-1)).float().to(self.device)

        # Combined reward
        total_rew = self.extrinsic_coef * ext + self.intrinsic_coef * int_norm

        # Current Q-values
        q_vals = self.q_net(obs)
        q_val = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_q_vals = self.q_net(next_obs)
            next_actions = next_q_vals.argmax(1)
            target_q_vals = self.target_q_net(next_obs)
            max_next_q = target_q_vals.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = total_rew + (1 - done) * self.gamma * max_next_q

        # Huber loss for stability
        loss = F.smooth_l1_loss(q_val, target)
        
        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

    def plot_intrinsic_vs_true_bonus_heatmap_minigrid_rnd(self, step):
        """
        plots true count bonus vs rnd intrinsic bonus on the minigrid.
        faithful to rnd: bonus = mse(pred-target) on whitened+clipped obs, then
        normalized by running std of discounted intrinsic returns.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import wandb
        import torch
        try:
            from minigrid.core.world_object import Wall
        except Exception:
            Wall = None

        visit_counts = self.visit_counts
        h, w = visit_counts.shape
        true_bonus = 1.0 / np.sqrt(visit_counts + 1e-8)
        rnd_bonus = np.full_like(true_bonus, np.nan, dtype=np.float32)

        base_env = self.eval_env.unwrapped
        try:
            base_env.tile_size = 4  # match RGBImgObsWrapper(tile_size=4)
        except Exception:
            pass
        
        # Get reward standard deviation for normalization
        reward_std = np.sqrt(self.reward_rms.var + 1e-8)

        for y in range(h):
            for x in range(w):
                # skip walls if we can detect them
                try:
                    cell = base_env.grid.get(x + 1, y + 1)
                    if Wall is not None and isinstance(cell, Wall):
                        continue
                except Exception:
                    pass

                try:
                    base_env.reset()
                    base_env.agent_pos = (x + 1, y + 1)
                    base_env.agent_dir = np.random.randint(0, 4)

                    base_env.step(base_env.actions.toggle)

                    obs_raw = base_env.render()
                    obs = self.process_obs(obs_raw)
                    obs_tensor = self.obs_to_float_tensor(obs)

                    with torch.no_grad():
                        bonus_raw = float(self.compute_intrinsic_reward(obs_tensor).item())

                    if np.isfinite(bonus_raw) and reward_std > 0:
                        rnd_bonus[y, x] = bonus_raw / reward_std
                except Exception:
                    continue

        # Create visualization
        true_mask = visit_counts == 0
        true_bonus_masked = np.ma.array(true_bonus, mask=true_mask)

        finite = np.isfinite(rnd_bonus)
        if finite.any():
            vals = rnd_bonus[finite]
            vmin = np.nanpercentile(vals, 1)
            vmax = np.nanpercentile(vals, 99)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        else:
            vmin, vmax = 0.0, 1.0
        rnd_img = np.ma.masked_invalid(rnd_bonus)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        im0 = axs[0].imshow(true_bonus_masked, cmap="magma",
                            vmin=np.nanmin(true_bonus_masked),
                            vmax=np.nanmax(true_bonus_masked))
        axs[0].set_title("true bonus (1/sqrt(count))")
        for yy in range(h):
            for xx in range(w):
                if visit_counts[yy, xx] > 0:
                    axs[0].text(xx, yy, f"{visit_counts[yy, xx]}",
                                ha="center", va="center",
                                color="white" if true_bonus[yy, xx] <
                                (true_bonus_masked.min() + true_bonus_masked.max()) / 2 else "black")
        fig.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(rnd_img, cmap="viridis", vmin=vmin, vmax=vmax)
        axs[1].set_title("rnd intrinsic bonus")
        fig.colorbar(im1, ax=axs[1])

        for ax in axs:
            ax.set_xticks(range(w))
            ax.set_yticks(range(h))

        plt.tight_layout()
        wandb.log({"heatmap/true_vs_intrinsic_bonus_heatmap": wandb.Image(fig)}, step=step)
        plt.close(fig)

    def print_visit_counts(self):
        print("\nTrue visit counts:")
        for row in self.visit_counts:
            print(" ".join(f"{v:4d}" for v in row))

    def log_action_gap(self, step, num_samples=500):
        """
        Samples states from replay buffer and logs action gap stats to wandb.
        """
        if self.replay_buffer.get_stored_size() < num_samples:
            return

        batch = self.replay_buffer.sample(num_samples)
        obs_batch = self.obs_to_float_tensor(batch["obs"])

        with torch.no_grad():
            q_vals = self.q_net(obs_batch)

        max_q, argmax_a = q_vals.max(dim=1)
        q_vals_clone = q_vals.clone()
        q_vals_clone[torch.arange(q_vals.size(0)), argmax_a] = -1e10
        second_best_q = q_vals_clone.max(dim=1)[0]

        gaps = (max_q - second_best_q).cpu().numpy()

        wandb.log({
            "diagnostics/avg_action_gap": np.mean(gaps),
            "diagnostics/action_gap_hist": wandb.Histogram(gaps)
        }, step=step)


def main():
    wandb.init(project="dqn", name="rnd_fixed_v2")
    ENV_NAME = "Fixed-DoorKey-6x6-v0"

    # train env
    env = gym.make(
        "Fixed-DoorKey-v0",
        size=10,
        key_pos=(1, 8),
        door_pos=(5, 5),
        goal_pos=(8, 5),
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        disable_env_checker=True,
        max_episode_steps=400,
        render_mode="rgb_array",
    )
    env = customised_doorkey.NoDropWrapper(env)
    env = FullyObsWrapper(env)
    env = RGBImgObsWrapper(env, tile_size=4)
    env = ImgObsWrapper(env)

    # eval env
    eval_env = gym.make(
        "Fixed-DoorKey-v0",
        size=10,
        key_pos=(1, 8),
        door_pos=(5, 5),
        goal_pos=(8, 5),
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        disable_env_checker=True,
        max_episode_steps=400,
        render_mode="rgb_array",
    )
    eval_env = customised_doorkey.NoDropWrapper(eval_env)
    eval_env = FullyObsWrapper(eval_env)
    eval_env = RGBImgObsWrapper(eval_env, tile_size=4)
    eval_env = ImgObsWrapper(eval_env)

    max_episode_steps = 400
    total_timesteps = 1_000_000

    # Adjusted hyperparameters based on RND paper
    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": 256,  
        "lr": 1e-4,
        "gamma": 0.99, 
        "batch_size": 128,  
        "replay_buffer_size": 1_000_000,
        "target_update_freq": 1000,  
        "train_every": 4,  
    })

    rnd_cfg = type("RNDConfig", (), {
        "intrinsic_coef": 1.0,
        "extrinsic_coef": 2.0,
        "lr": 1e-4,
        "learning_starts": 5_000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05, 
        "epsilon_decay": 0.9995,  
        "predictor_keep_ratio": 0.25,
    })

    agent = DQN_RNDAgent(env=env, env_name=ENV_NAME, eval_env=eval_env, dqn_cfg=dqn_cfg, rnd_cfg=rnd_cfg)

    start = time.time()
    agent.train(total_timesteps=total_timesteps, max_episode_steps=max_episode_steps)
    end = time.time()
    print(f"Training finished in {(end - start) / 60:.2f} minutes.")
    agent.print_visit_counts()
  

if __name__ == "__main__":
    main()