import sys
import os

from matplotlib import pyplot as plt

from RND.rnd import RNDModel, RewardForwardFilter
from agents.dqn_cfn import evaluate_dqn
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
from gymnasium.wrappers.utils import RunningMeanStd
from minigrid.wrappers import FullyObsWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


class DQN_RNDAgent:
    def __init__(self, env, eval_env, dqn_cfg, rnd_cfg, env_name):
        self.env = env
        self.eval_env = eval_env
        self.env_name = env_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        act_dim = env.action_space.n

        H_all = self.env.unwrapped.grid.height
        W_all = self.env.unwrapped.grid.width
        H_in, W_in = H_all - 2, W_all - 2
        self.visit_counts = np.zeros((H_in, W_in), dtype=np.int32)

        self.N_TYPE, self.N_COLOR, self.N_STATE = 11, 6, 4
        self.FEAT_PER_CELL = self.N_TYPE + self.N_COLOR + self.N_STATE

        self._eye_type = np.eye(self.N_TYPE, dtype=np.float32)
        self._eye_color = np.eye(self.N_COLOR, dtype=np.float32)
        self._eye_state = np.eye(self.N_STATE, dtype=np.float32)

        self.obs_dim = H_in * W_in * self.FEAT_PER_CELL + 2 + 4

        self.q_net = DDQN(self.obs_dim, act_dim, dqn_cfg.hidden_size, is_cnn=False).to(self.device)
        self.target_q_net = DDQN(self.obs_dim, act_dim, dqn_cfg.hidden_size, is_cnn=False).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=dqn_cfg.lr)
        self.gamma = dqn_cfg.gamma
        self.batch_size = dqn_cfg.batch_size
        self.target_update_freq = dqn_cfg.target_update_freq

        self.replay_buffer = ReplayBuffer(
            dqn_cfg.replay_buffer_size,
            env_dict={
                "obs":      {"shape": (self.obs_dim,), "dtype": np.float32},
                "act":      {"shape": 1, "dtype": np.int16},
                "ext_rew":  {"shape": 1, "dtype": np.float32},
                "int_rew":  {"shape": 1, "dtype": np.float32},
                "term":     {"shape": 1, "dtype": np.bool_},   
                "timeout":  {"shape": 1, "dtype": np.bool_},   
                "next_obs": {"shape": (self.obs_dim,), "dtype": np.float32},
            },
        )


        self.rnd_cfg = rnd_cfg
        self.intrinsic_coef = rnd_cfg.intrinsic_coef
        self.extrinsic_coef = rnd_cfg.extrinsic_coef

        self.rnd = RNDModel(self.obs_dim, output_size=512).to(self.device)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_cfg.lr)

        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))
        self.reward_rms = RunningMeanStd(shape=())
        self.reward_filter = RewardForwardFilter(gamma=0.99)

    def process_obs(self, obs_raw, env_for_pose=None):
        img_full = np.asarray(obs_raw["image"], dtype=np.int16)
        img = img_full[1:-1, 1:-1, :]
        H_in, W_in = img.shape[:2]

        t = img[..., 0].clip(0, self.N_TYPE - 1)
        c = img[..., 1].clip(0, self.N_COLOR - 1)
        s = img[..., 2].clip(0, self.N_STATE - 1)

        oh_t = self._eye_type[t]
        oh_c = self._eye_color[c]
        oh_s = self._eye_state[s]
        cell_feats = np.concatenate([oh_t, oh_c, oh_s], axis=-1).reshape(-1).astype(np.float32)

        env0 = env_for_pose if env_for_pose is not None else self.env
        ax, ay = map(int, env0.unwrapped.agent_pos)
        ax0 = (ax - 1) / max(W_in - 1, 1)
        ay0 = (ay - 1) / max(H_in - 1, 1)

        d = int(getattr(env0.unwrapped, "agent_dir", 0)) % 4
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[d] = 1.0

        return np.concatenate([cell_feats, np.array([ax0, ay0], np.float32), dir_onehot], 0)

    def increment_visit_counts(self):
        x, y = map(int, self.env.unwrapped.agent_pos)
        self.visit_counts[(y - 1), (x - 1)] += 1

    def act(self, obs, epsilon):
        if isinstance(obs, np.ndarray):
            obs_t = torch.from_numpy(obs).to(self.device).float()
        else:
            obs_t = obs.to(self.device).float()
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)            # (1, D)

        with torch.no_grad():
            q = self.q_net(obs_t)                 # (B, A)

        greedy = q.argmax(dim=1).cpu().numpy().astype(int)  # (B,)

        if epsilon > 0.0:
            B = greedy.shape[0]
            mask = np.random.rand(B) < float(epsilon)
            if mask.any():
                greedy[mask] = np.random.randint(0, self.env.action_space.n, size=mask.sum())

        return int(greedy[0]) if greedy.shape[0] == 1 else greedy


    def obs_to_float_tensor(self, obs):
        if isinstance(obs, np.ndarray):
            t = torch.from_numpy(obs).to(self.device)
        else:
            t = obs.to(self.device)
        return t.float()

    def update_obs_rms(self, obs_vec: np.ndarray):
        self.obs_rms.update(obs_vec[None, :])

    def normalize_obs_with_rms(self, obs_tensor):
        obs_mean = torch.from_numpy(self.obs_rms.mean).float().to(self.device)
        obs_std = torch.sqrt(torch.from_numpy(self.obs_rms.var).float().to(self.device) + 1e-8)
        return (obs_tensor - obs_mean) / obs_std

    def compute_intrinsic_reward(self, obs_tensor):
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        obs_mean = torch.from_numpy(self.obs_rms.mean).float().to(self.device)
        obs_var = torch.from_numpy(self.obs_rms.var).float().to(self.device)
        normalized_obs = (obs_tensor - obs_mean) / torch.sqrt(obs_var + 1e-8)
        normalized_obs = torch.clamp(normalized_obs, -5.0, 5.0)
        with torch.no_grad():
            pred, target = self.rnd(normalized_obs)
            intrinsic = 0.5 * ((pred - target) ** 2).sum(dim=1)
        return intrinsic.squeeze(0)

    def _fit_obs_rms_warmup(self, frames=20000, max_ep_len=200):
        seen = 0
        while seen < frames:
            obs_raw, _ = self.env.reset()
            done, steps = False, 0
            while not done and steps < max_ep_len and seen < frames:
                obs_vec = self.process_obs(obs_raw)
                self.obs_rms.update(obs_vec[None, :])
                a = self.env.action_space.sample()
                obs_raw, _, term, trunc, _ = self.env.step(a)
                done = term or trunc
                steps += 1
                seen += 1

    def train(self, total_timesteps, max_episode_steps):
        current_timestep = 0
        episode_return = 0
        episode_step = 0
        episode_num = 0

        obs_raw, _ = self.env.reset()
        obs = self.process_obs(obs_raw)
        self.update_obs_rms(obs)

        stats = EpisodeStats([], [], [])
        epsilon = self.rnd_cfg.epsilon_start

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

            done = bool(terminated or truncated)

            term = bool(terminated)
            timeout = bool(truncated)

            if ext_reward > 0:
                wandb.log({"ext_rew": ext_reward}, step=current_timestep)

            next_obs_tensor = self.obs_to_float_tensor(next_obs).unsqueeze(0)
            int_reward_tensor = self.compute_intrinsic_reward(next_obs_tensor)
            int_reward = float(int_reward_tensor.item())

            discounted_r = self.reward_filter.update(int_reward)
            self.reward_rms.update(np.array([discounted_r]))
            norm_int_reward = int_reward / np.sqrt(np.maximum(self.reward_rms.var, 1e-8))

            if current_timestep % 1000 == 0:
                agent_x, agent_y = map(int, self.env.unwrapped.agent_pos)
                wandb.log({f"int_reward/cell({agent_y - 1},{agent_x - 1})": norm_int_reward}, step=current_timestep)

            total_reward = self.extrinsic_coef * ext_reward + self.intrinsic_coef * norm_int_reward

            self.replay_buffer.add(
                obs=obs,
                act=action,
                ext_rew=ext_reward,
                int_rew=norm_int_reward,
                term=term,
                timeout=timeout,
                next_obs=next_obs,
            )

            if current_timestep > self.rnd_cfg.learning_starts and self.replay_buffer.get_stored_size() >= self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)

                obs_batch = self.obs_to_float_tensor(batch["obs"])

                obs_mean = torch.from_numpy(self.obs_rms.mean).float().to(self.device)
                obs_var = torch.from_numpy(self.obs_rms.var).float().to(self.device)
                normalized = (obs_batch - obs_mean) / torch.sqrt(obs_var + 1e-8)
                normalized = torch.clamp(normalized, -5.0, 5.0)
                
                pred, target = self.rnd(normalized)
                
                per_sample = F.mse_loss(pred, target, reduction="none").mean(dim=1)
                keep = getattr(self.rnd_cfg, "predictor_keep_ratio", getattr(self.rnd_cfg, "rnd_mask_prob", 0.25))
                mask = (torch.rand_like(per_sample) < keep).float()
                if mask.sum() == 0:
                    mask = torch.ones_like(per_sample)
                loss = (per_sample * mask).sum() / mask.sum()
                
                self.rnd_optimizer.zero_grad()
                loss.backward()
                self.rnd_optimizer.step()

                batch = self.replay_buffer.sample(self.batch_size)
                self.update_dqn(batch)

            if current_timestep % 1000 == 0 and current_timestep > 0:
                wandb.log({
                    "a/ext_reward": ext_reward,
                    "a/int_reward": int_reward,
                    "a/epsilon": epsilon,
                }, step=current_timestep)

            self.update_obs_rms(next_obs)
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
                self.update_obs_rms(obs)
                self.increment_visit_counts()
                episode_return = 0
                episode_step = 0
                episode_num += 1

            if current_timestep % 10000 == 0 and current_timestep > 0:
                # validate_dqn(self, current_timestep, save_dir="checkpoints_rnd")
                evaluate_dqn(self, self.eval_env, current_timestep, save_dir="eval_rnd")

            if current_timestep % 5000 == 0:
                plot_intrinsic_vs_true_bonus_heatmap_minigrid_rnd(self, step=current_timestep)
                

    def update_dqn(self, batch):
        obs = self.obs_to_float_tensor(batch["obs"])
        next_obs = self.obs_to_float_tensor(batch["next_obs"])
        act = torch.from_numpy(batch["act"].squeeze(-1)).long().to(self.device)
        ext = torch.from_numpy(batch["ext_rew"].squeeze(-1)).float().to(self.device)
        int_norm = torch.from_numpy(batch["int_rew"].squeeze(-1)).float().to(self.device)
        term     = torch.from_numpy(batch["term"].squeeze(-1)).float().to(self.device)
        timeout  = torch.from_numpy(batch["timeout"].squeeze(-1)).float().to(self.device)


        total_rew = self.extrinsic_coef * ext + self.intrinsic_coef * int_norm

        q_vals = self.q_net(obs)
        q_val = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.q_net(next_obs)
            next_a = next_q.argmax(1)
            tgt_q = self.target_q_net(next_obs)
            max_next = tgt_q.gather(1, next_a.unsqueeze(1)).squeeze(1)
            bootstrap_mask = 1.0 - term  
            target = total_rew + bootstrap_mask * self.gamma * max_next

        loss = F.smooth_l1_loss(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def print_visit_counts(self):
        print("\nTrue visit counts:")
        for row in self.visit_counts:
            print(" ".join(f"{v:4d}" for v in row))


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



def main():
    wandb.init(project="dqn", name="rnd")
    ENV_NAME = "Fixed-DoorKey-v0"

    max_episode_steps = 400
    total_timesteps = 1_000_000

    env = gym.make(
        "Fixed-DoorKey-v0", size=10,
        key_pos=(1, 8), door_pos=(5, 5), goal_pos=(8, 1),
        agent_start_pos=(1, 1), agent_start_dir=0,
        disable_env_checker=True, max_episode_steps=max_episode_steps, render_mode="rgb_array",
    )
    env = customised_doorkey.PatchGridWrapper(env, wall_cells=[(6, 1), (7, 1)], goal_cell=(7, 0))
    env = FullyObsWrapper(env)
    env = customised_doorkey.NoDropWrapper(env)
    

    eval_env = gym.make(
        "Fixed-DoorKey-v0", size=10,
        key_pos=(1, 8), door_pos=(5, 5), goal_pos=(8, 1),
        agent_start_pos=(1, 1), agent_start_dir=0,
        disable_env_checker=True, max_episode_steps=max_episode_steps, render_mode="rgb_array",
    )
    eval_env = customised_doorkey.PatchGridWrapper(eval_env, wall_cells=[(6, 1), (7, 1)], goal_cell=(7, 0))
    eval_env = FullyObsWrapper(eval_env)
    eval_env = customised_doorkey.NoDropWrapper(eval_env)

    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": 512,
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "replay_buffer_size": 1_000_000,
        "target_update_freq": 2000
    })

    rnd_cfg = type("RNDConfig", (), {
        "intrinsic_coef": 1.0,
        "extrinsic_coef": 2.0,
        "lr": 1e-4,
        "learning_starts": 1000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.9998,
        "rnd_mask_prob": 0.25
    })

    agent = DQN_RNDAgent(
        env=env,
        env_name=ENV_NAME,
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
