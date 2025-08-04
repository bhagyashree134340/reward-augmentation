import sys
import os

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

from networks.ddqn import DDQN
from utils.stats import EpisodeStats
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


class RunningMeanStd:
    """Running mean and standard deviation calculation"""
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
    """Reward forward filter for normalizing intrinsic rewards"""
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
    """Random Network Distillation Model for CNN observations"""
    def __init__(self, obs_shape, hidden_size=512):
        super(RNDModel, self).__init__()
        
        c, h, w = obs_shape
        
        # MiniGrid-specific CNN architecture (same as your DDQN)
        # Target network (frozen, random weights)
        self.target = nn.Sequential(
            self._layer_init(nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
            self._layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
            self._layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # Always reduce to 2x2
            nn.Flatten(),
        )
        
        # Feature size is always 64 * 2 * 2 = 256 for MiniGrid
        conv_output_size = 64 * 2 * 2
        
        # Add final layers to target
        self.target.add_module('fc', self._layer_init(nn.Linear(conv_output_size, hidden_size)))
        
        # Predictor network (trainable) - same architecture
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
        
        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, gain=std)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, obs):
        # Normalize observations to [0, 1]
        if obs.dtype == torch.uint8:
            obs = obs.float() / 255.0
        elif obs.max() > 1.0:
            obs = obs / 255.0
            
        target_output = self.target(obs)
        predictor_output = self.predictor(obs)
        return predictor_output, target_output


class DQN_RNDAgent:
    def __init__(self, env, eval_env, dqn_cfg, rnd_cfg, env_name):
        self.env = env
        self.eval_env = eval_env
        self.env_name = env_name  # Store environment name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get observation shape from environment
        self.obs_shape = env.observation_space.shape  # Should be (H, W, C) from MiniGrid
        act_dim = env.action_space.n

        print("Original obs shape:", env.observation_space.shape)
        
        # Convert to (C, H, W) format for PyTorch
        if len(self.obs_shape) == 3:
            self.obs_shape_torch = (self.obs_shape[2], self.obs_shape[0], self.obs_shape[1])  # (C, H, W)
        else:
            raise ValueError(f"Unexpected observation shape: {self.obs_shape}")
            
        print("PyTorch obs shape:", self.obs_shape_torch)

        # Initialize DQN networks
        self.q_net = DDQN(self.obs_shape_torch, act_dim, dqn_cfg.hidden_size, is_cnn=True).to(self.device)
        self.target_q_net = DDQN(self.obs_shape_torch, act_dim, dqn_cfg.hidden_size, is_cnn=True).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=dqn_cfg.lr)
        self.gamma = dqn_cfg.gamma
        self.batch_size = dqn_cfg.batch_size
        self.target_update_freq = dqn_cfg.target_update_freq

        self.replay_buffer = deque(maxlen=dqn_cfg.replay_buffer_size)

        # Initialize RND components
        self.rnd_cfg = rnd_cfg
        self.intrinsic_coef = rnd_cfg.intrinsic_coef
        self.extrinsic_coef = rnd_cfg.extrinsic_coef

        self.rnd = RNDModel(self.obs_shape_torch).to(self.device)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_cfg.lr)

        # Running statistics for normalization
        obs_dim = np.prod(self.obs_shape_torch)
        self.obs_rms = RunningMeanStd(shape=self.obs_shape_torch)
        self.reward_rms = RunningMeanStd()
        self.reward_filter = RewardForwardFilter(gamma=0.99)

    def process_obs(self, obs):
        """Convert observation from (H, W, C) to (C, H, W) format"""
        if isinstance(obs, dict) and 'image' in obs:
            # Handle dict observation (some MiniGrid versions return dict)
            obs = obs['image']
        
        # Ensure it's a numpy array
        obs = np.array(obs, dtype=np.uint8)
        
        # Convert from (H, W, C) to (C, H, W)
        if len(obs.shape) == 3:
            obs = np.transpose(obs, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")
            
        return obs

    def act(self, obs, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return q_values.argmax().item()

    def compute_intrinsic_reward(self, obs_tensor):
        """Compute RND intrinsic reward"""
        with torch.no_grad():
            # Update observation statistics
            obs_np = obs_tensor.cpu().numpy()
            self.obs_rms.update(obs_np)
            
            # Normalize observation
            obs_mean = torch.from_numpy(self.obs_rms.mean).float().to(self.device)
            obs_std = torch.sqrt(torch.from_numpy(self.obs_rms.var).float().to(self.device) + 1e-8)
            norm_obs = (obs_tensor - obs_mean) / obs_std
            
            # Compute RND prediction error
            pred, target = self.rnd(norm_obs)
            intrinsic_reward = 0.5 * ((pred - target) ** 2).sum(dim=1)
            
            return intrinsic_reward

    def train(self, total_timesteps, max_episode_steps):
        current_timestep = 0
        episode_return = 0
        episode_step = 0
        episode_num = 0
        obs_raw, _ = self.env.reset()
        obs = self.process_obs(obs_raw)
        avg_rewards = []
        avg_int_rewards = []
        avg_ext_rewards = []

        stats = EpisodeStats([], [], [])

        epsilon = self.rnd_cfg.epsilon_start

        while current_timestep < total_timesteps:
            # epsilon = max(self.rnd_cfg.epsilon_end, epsilon * self.rnd_cfg.epsilon_decay)
            action = self.act(obs, epsilon)

            next_obs_raw, ext_reward, terminated, truncated, _ = self.env.step(action)
            next_obs = self.process_obs(next_obs_raw)

            done = truncated or terminated

            # Compute intrinsic reward using RND
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            int_reward_tensor = self.compute_intrinsic_reward(next_obs_tensor)
            int_reward = int_reward_tensor.item()

            # Normalize intrinsic reward
            discounted_r = self.reward_filter.update(int_reward)
            self.reward_rms.update(np.array([discounted_r]))
            norm_int_reward = int_reward / np.sqrt(np.maximum(self.reward_rms.var, 1e-8))

            # Combine rewards
            total_reward = self.extrinsic_coef * ext_reward + self.intrinsic_coef * norm_int_reward
            
            avg_rewards.append(total_reward)
            avg_int_rewards.append(norm_int_reward)
            avg_ext_rewards.append(ext_reward)

            # Store transition in replay buffer
            self.replay_buffer.append((obs, action, ext_reward, norm_int_reward, next_obs, done))

            # Train RND network
            mask_prob = getattr(self.rnd_cfg, 'rnd_mask_prob', 0.25)
            if current_timestep > self.rnd_cfg.learning_starts:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Update observation statistics and normalize
                obs_np = obs_tensor.cpu().numpy()
                self.obs_rms.update(obs_np)
                obs_mean = torch.from_numpy(self.obs_rms.mean).float().to(self.device)
                obs_std = torch.sqrt(torch.from_numpy(self.obs_rms.var).float().to(self.device) + 1e-8)
                norm_obs = (obs_tensor - obs_mean) / obs_std
                
                pred, target = self.rnd(norm_obs)
                mask = torch.rand_like(pred[:, 0]) < mask_prob
                if mask.sum() > 0:
                    forward_loss = F.mse_loss(pred[mask], target.detach()[mask])
                    self.rnd_optimizer.zero_grad()
                    forward_loss.backward()
                    self.rnd_optimizer.step()

            # Train DQN
            if current_timestep > self.rnd_cfg.learning_starts and len(self.replay_buffer) >= self.batch_size:
                batch = random.sample(self.replay_buffer, self.batch_size)
                self.update_dqn(batch)

            # Log metrics
            if current_timestep % 1000 == 0 and current_timestep > 0:
                wandb.log({
                    "ext_reward": np.mean(avg_ext_rewards) if avg_ext_rewards else 0,
                    "int_reward": np.mean(avg_int_rewards) if avg_int_rewards else 0,
                    "total_reward": np.mean(avg_rewards) if avg_rewards else 0,
                    "epsilon": epsilon,
                }, step=current_timestep)
                avg_rewards.clear()
                avg_int_rewards.clear()
                avg_ext_rewards.clear()

            obs = next_obs
            episode_return += total_reward
            episode_step += 1
            current_timestep += 1

            # Update target network
            if current_timestep % self.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            # Evaluation during training
            if current_timestep % 50000 == 0 and current_timestep > 0:
                print(f"\n--- Evaluation at step {current_timestep} ---")

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

                if epsilon > self.cfn_cfg.epsilon_end:
                    epsilon *= self.cfn_cfg.epsilon_decay

                obs_raw, _ = self.env.reset()
                obs = self.process_obs(obs_raw)
                episode_return = 0
                episode_step = 0
                episode_num += 1

            if current_timestep % 10000 == 0 and current_timestep > 0:
                    validate_dqn(self, current_timestep, save_dir="checkpoints_rnd")
                    evaluate_dqn(self, self.env, current_timestep, save_dir="eval_rnd")

    def update_dqn(self, batch):
        """Update DQN networks"""
        obs, act, ext_rew, int_rew, next_obs, done = map(np.array, zip(*batch))
        
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        act = torch.tensor(act, dtype=torch.long, device=self.device)
        ext_rew = torch.tensor(ext_rew, dtype=torch.float32, device=self.device)
        int_rew = torch.tensor(int_rew, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        # Combine rewards
        total_rew = self.extrinsic_coef * ext_rew + self.intrinsic_coef * int_rew

        # Current Q values
        q_vals = self.q_net(obs)
        q_val = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q_vals = self.q_net(next_obs)
            next_actions = next_q_vals.argmax(1)

            target_q_vals = self.target_q_net(next_obs)
            max_next_q_vals = target_q_vals.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target = total_rew + (1 - done) * self.gamma * max_next_q_vals

        # Compute loss and update
        loss = F.mse_loss(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

def main():
    # Initialize wandb
    wandb.init(project="dqn", name="rnd")
    ENV_NAME = "MiniGrid-DoorKey-6x6-v0"  

    # Create environment
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    eval_env = gym.make(ENV_NAME, render_mode="rgb_array")
    eval_env = FullyObsWrapper(eval_env)
    eval_env = ImgObsWrapper(eval_env)
    
    max_episode_steps = 250
    total_timesteps = 1300_000

    # DQN hyperparameters
    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": 128,
        "lr": 5e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "replay_buffer_size": 50_000,
        "target_update_freq": 1000
    })

    # RND hyperparameters
    rnd_cfg = type("RNDConfig", (), {
        "intrinsic_coef": 1.0,
        "extrinsic_coef": 2.0,
        "lr": 1e-4,
        "learning_starts": 10000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.9998,
        "rnd_mask_prob": 0.25
    })

    # Create agent
    agent = DQN_RNDAgent(
        env=env,
        env_name = ENV_NAME,
        eval_env=eval_env,
        dqn_cfg=dqn_cfg,
        rnd_cfg=rnd_cfg
    )

    # Train agent
    start = time.time()
    agent.train(total_timesteps=total_timesteps, max_episode_steps=max_episode_steps)
    end = time.time()
    print(f"Training finished in {(end - start) / 60:.2f} minutes.")

if __name__ == "__main__":
    main()