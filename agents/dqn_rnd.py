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

        # self.replay_buffer = deque(maxlen=dqn_cfg.replay_buffer_size)
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

        obs_dim = np.prod(self.obs_shape_torch)
        self.obs_rms = RunningMeanStd(shape=self.obs_shape_torch)
        self.reward_rms = RunningMeanStd()
        self.reward_filter = RewardForwardFilter(gamma=0.99)

    def _agent_rc(self):
        x, y = map(int, self.env.unwrapped.agent_pos)
        return (y - 1), (x - 1)
    
    def build_episode_intrinsic_map(self):
        """Run ONE eval episode, collect mean normalized intrinsic reward per interior cell."""
        H, W = self.visit_counts.shape
        ep_sum = np.zeros((H, W), dtype=np.float64)
        ep_cnt = np.zeros((H, W), dtype=np.int32)

        obs_raw, _ = self.eval_env.reset()
        obs = self.process_obs(obs_raw)

        done = False
        steps = 0
        # ε-greedy with mild exploration so we visit more cells
        epsilon_eval = 0.2

        while not done and steps < 1000:
            # compute RND int reward for CURRENT obs (or next_obs — your choice; keep consistent)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            # normalize like in training
            obs_np = obs_tensor.cpu().numpy()
            self.obs_rms.update(obs_np)  # optional: comment out if you want *frozen* stats
            obs_mean = torch.from_numpy(self.obs_rms.mean).float().to(self.device)
            obs_std  = torch.sqrt(torch.from_numpy(self.obs_rms.var).float().to(self.device) + 1e-8)
            norm_obs = (obs_tensor - obs_mean) / obs_std

            with torch.no_grad():
                pred, tgt = self.rnd(norm_obs)
                bonus = 0.5 * ((pred - tgt) ** 2).sum().item()

            norm_int_reward = bonus / np.sqrt(self.reward_rms.var + 1e-8)

            # bin to interior cell
            ry, rx = self._agent_rc()
            ep_sum[ry, rx] += norm_int_reward
            ep_cnt[ry, rx] += 1

            # act
            if np.random.rand() < epsilon_eval:
                action = self.eval_env.action_space.sample()
            else:
                with torch.no_grad():
                    q = self.q_net(obs_tensor / 255.0)
                    action = q.argmax(1).item()

            next_obs_raw, _, terminated, truncated, _ = self.eval_env.step(action)
            obs = self.process_obs(next_obs_raw)
            done = terminated or truncated
            steps += 1

        denom = np.maximum(1, ep_cnt)
        ep_mean_intrinsic = ep_sum / denom
        return ep_mean_intrinsic, ep_cnt

    def plot_intrinsic_vs_true_bonus_heatmap_minigrid(self, current_timestep, save_dir="plots"):
        H, W = self.visit_counts.shape
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        out = save_path / f"minigrid_rnd_vs_true_bonus_{H}x{W}_{current_timestep}.png"

        # 1) Build episode map
        ep_mean_intrinsic, ep_cnt = self.build_episode_intrinsic_map()

        # 2) Global true bonus from training counts
        true_counts = self.visit_counts
        true_bonus = 1.0 / np.sqrt(true_counts + 1e-8)

        visited_mask = (true_counts > 0)
        masked_true_bonus = true_bonus[visited_mask]
        # masked_rnd_bonus  = ep_mean_intrinsic[visited_mask]

        # 3) Plot
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        cmaps  = ["magma", "magma"]
        titles = ["True Bonus (1/sqrt(N))", "RND Bonus (mean per cell, ep)"]

        for ax, data, title, cmap in zip(axs[:2], [true_bonus, ep_mean_intrinsic], titles, cmaps):
            im = ax.imshow(data, cmap=cmap, interpolation="nearest")
            ax.set_title(title)
            ax.set_xticks(range(W)); ax.set_yticks(range(H))
            fig.colorbar(im, ax=ax)

        axs[2].scatter(masked_true_bonus.ravel(), ep_mean_intrinsic.ravel(), s=12)
        axs[2].set_xlabel("True Bonus (1/sqrt(N))")
        axs[2].set_ylabel("RND Bonus (normalized)")
        axs[2].set_title("True vs. Approx Bonus")
        axs[2].grid(True)

        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print(f"Heatmap and scatter plot saved to {out}")

    def process_obs(self, obs):
        if isinstance(obs, dict) and 'image' in obs:
            obs = obs['image']
        
        obs = np.array(obs, dtype=np.uint8)
        
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
        with torch.no_grad():
            obs_np = obs_tensor.cpu().numpy()
            self.obs_rms.update(obs_np)
            
            obs_mean = torch.from_numpy(self.obs_rms.mean).float().to(self.device)
            obs_std = torch.sqrt(torch.from_numpy(self.obs_rms.var).float().to(self.device) + 1e-8)
            norm_obs = (obs_tensor - obs_mean) / obs_std
            
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

        ry, rx = self._agent_rc()
        self.visit_counts[ry, rx] += 1
        avg_rewards = []
        avg_int_rewards = []
        avg_ext_rewards = []

        stats = EpisodeStats([], [], [])

        epsilon = self.rnd_cfg.epsilon_start

        while current_timestep < total_timesteps:
            action = self.act(obs, epsilon)

            next_obs_raw, ext_reward, terminated, truncated, _ = self.env.step(action)
            next_obs = self.process_obs(next_obs_raw)

            ry, rx = self._agent_rc()
            self.visit_counts[ry, rx] += 1

            done = truncated or terminated

            if ext_reward > 0:
                wandb.log({
                    "ext_rew": ext_reward
                }, step=current_timestep)

            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            int_reward_tensor = self.compute_intrinsic_reward(next_obs_tensor)
            int_reward = int_reward_tensor.item()

            discounted_r = self.reward_filter.update(int_reward)
            self.reward_rms.update(np.array([discounted_r]))
            norm_int_reward = int_reward / np.sqrt(np.maximum(self.reward_rms.var, 1e-8))

            total_reward = self.extrinsic_coef * ext_reward + self.intrinsic_coef * norm_int_reward
            
            avg_rewards.append(total_reward)
            avg_int_rewards.append(norm_int_reward)
            avg_ext_rewards.append(ext_reward)

            # self.replay_buffer.append((obs, action, ext_reward, norm_int_reward, next_obs, done))
            self.replay_buffer.add(
                obs=obs,
                act=action,
                ext_rew=ext_reward,
                int_rew=norm_int_reward,   
                done=done,
                next_obs=next_obs,
            )

            mask_prob = getattr(self.rnd_cfg, 'rnd_mask_prob', 0.25)
            if current_timestep > self.rnd_cfg.learning_starts:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                
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

            if current_timestep > self.rnd_cfg.learning_starts and self.replay_buffer.get_stored_size() >= self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                self.update_dqn(batch)

            if current_timestep % 1000 == 0 and current_timestep > 0:
                wandb.log({
                    "a/ext_reward": ext_reward,
                    "a/int_reward": int_reward,
                    "a/epsilon": epsilon,
                }, step=current_timestep)
                avg_rewards.clear()
                avg_int_rewards.clear()
                avg_ext_rewards.clear()

            obs = next_obs
            episode_return += total_reward
            episode_step += 1
            current_timestep += 1

            if current_timestep % self.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

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

                obs_raw, _ = self.env.reset()
                obs = self.process_obs(obs_raw)
                ry, rx = self._agent_rc()
                self.visit_counts[ry, rx] += 1
                episode_return = 0
                episode_step = 0
                episode_num += 1

            if current_timestep % 10000 == 0 and current_timestep > 0:
                validate_dqn(self, current_timestep, save_dir="checkpoints_rnd")
                evaluate_dqn(self, self.eval_env, current_timestep, save_dir="eval_rnd")

            if current_timestep % 50000 == 0:
                self.plot_intrinsic_vs_true_bonus_heatmap_minigrid(current_timestep)

    def update_dqn(self, batch):
        # cpprb returns numpy arrays with shapes:
        # obs: (B, C, H, W) uint8, next_obs: same
        # act/ext_rew/int_rew/done: (B, 1)
        obs      = torch.tensor(batch["obs"],      dtype=torch.float32, device=self.device) / 255.0
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device) / 255.0

        act      = torch.from_numpy(batch["act"].squeeze(-1)).long().to(self.device)
        ext_rew  = torch.from_numpy(batch["ext_rew"].squeeze(-1)).float().to(self.device)
        int_rew  = torch.from_numpy(batch["int_rew"].squeeze(-1)).float().to(self.device)
        done     = torch.from_numpy(batch["done"].squeeze(-1)).float().to(self.device)

        total_rew = self.extrinsic_coef * ext_rew + self.intrinsic_coef * int_rew

        q_vals = self.q_net(obs)
        q_val = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_vals = self.q_net(next_obs)
            next_actions = next_q_vals.argmax(1)
            target_q_vals = self.target_q_net(next_obs)
            max_next_q_vals = target_q_vals.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = total_rew + (1 - done) * self.gamma * max_next_q_vals

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

    # env = gym.make(ENV_NAME, render_mode="rgb_array")
    # env = FullyObsWrapper(env)
    # env = ImgObsWrapper(env)

    # eval_env = gym.make(ENV_NAME, render_mode="rgb_array")
    # eval_env = FullyObsWrapper(eval_env)
    # eval_env = ImgObsWrapper(eval_env)

    env = gym.make(
        "Fixed-DoorKey-v0",           # generic registration
        size=16,
        disable_env_checker=True,
        render_mode="rgb_array",
        key_pos=(1, 14),              # bottom-left interior
        door_pos=(8, 8),              # middle of vertical wall
        goal_pos=(14, 14),            # bottom-right interior
        agent_start_pos=(1, 1),       # top-left interior
    )
    env = customised_doorkey.NoDropWrapper(env)
    env = FullyObsWrapper(env)
    env = RGBImgObsWrapper(env, tile_size=4)
    env = ImgObsWrapper(env)

    eval_env = gym.make(
        "Fixed-DoorKey-v0",           # generic registration
        size=16,
        disable_env_checker=True,
        render_mode="rgb_array",
        key_pos=(1, 14),              # bottom-left interior
        door_pos=(8, 8),              # middle of vertical wall
        goal_pos=(14, 14),            # bottom-right interior
        agent_start_pos=(1, 1),       # top-left interior
    )
    eval_env = customised_doorkey.NoDropWrapper(eval_env)
    eval_env = FullyObsWrapper(eval_env)
    eval_env = RGBImgObsWrapper(eval_env, tile_size=4)
    eval_env = ImgObsWrapper(eval_env)
    
    max_episode_steps = 1800
    total_timesteps = 1_300_000

    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": 128,
        "lr": 1e-5,
        "gamma": 0.99,
        "batch_size": 128,
        "replay_buffer_size": 1_000_000,
        "target_update_freq": 2000
    })

    rnd_cfg = type("RNDConfig", (), {
        "intrinsic_coef": 1.0,
        "extrinsic_coef": 2.0,
        "lr": 1e-4,
        "learning_starts": 10000,
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
    # agent.plot_intrinsic_vs_true_bonus_heatmap_minigrid()

if __name__ == "__main__":
    main()