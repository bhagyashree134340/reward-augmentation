import sys
import os

from matplotlib import pyplot as plt

# --- removed: RND imports ---
# from RND.rnd import RNDModel, RewardForwardFilter
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
# --- removed: RunningMeanStd from gymnasium.wrappers.utils ---
from minigrid.wrappers import FullyObsWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


class DQN_MRL_Agent:
    """
    DQN with Munchausen RL:
      r̄_t = r_t + α * clip(log π(a_t|s_t), lo, 0)
      V̄(s') = Σ_a' π(a'|s') [ Q(s',a') - τ log π(a'|s') ]
      target = r̄_t + γ * 1_{not terminal} * V̄(s')
    """
    def __init__(self, env, eval_env, dqn_cfg, mrl_cfg, env_name):
        self.env = env
        self.eval_env = eval_env
        self.env_name = env_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = dqn_cfg.gamma
        self.batch_size = dqn_cfg.batch_size
        self.target_update_freq = dqn_cfg.target_update_freq
        self.lr = dqn_cfg.lr

        # Munchausen hyperparams
        self.tau = float(mrl_cfg.tau)           # entropy temperature
        self.alpha_m = float(mrl_cfg.alpha_m)   # Munchausen coefficient
        self.lo = float(mrl_cfg.lo)             # clipping lower bound for log π
        self.extrinsic_coef = float(getattr(mrl_cfg, "extrinsic_coef", 1.0))  # for logging/return scale only

        act_dim = env.action_space.n

        # --- your symbolic full-grid → vector encoding stays the same ---
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

        # Q networks
        self.q_net = DDQN(self.obs_dim, act_dim, dqn_cfg.hidden_size, is_cnn=False).to(self.device)
        self.target_q_net = DDQN(self.obs_dim, act_dim, dqn_cfg.hidden_size, is_cnn=False).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        # Replay buffer (no intrinsic fields)
        self.replay_buffer = ReplayBuffer(
            dqn_cfg.replay_buffer_size,
            env_dict={
                "obs":      {"shape": (self.obs_dim,), "dtype": np.float32},
                "act":      {"shape": 1, "dtype": np.int16},
                "ext_rew":  {"shape": 1, "dtype": np.float32},
                "term":     {"shape": 1, "dtype": np.bool_},
                "timeout":  {"shape": 1, "dtype": np.bool_},
                "next_obs": {"shape": (self.obs_dim,), "dtype": np.float32},
            },
        )

    # ---------- encoding stays unchanged ----------
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

    # ---------- epsilon-greedy ----------
    def act(self, obs, epsilon):
        obs_t = torch.from_numpy(obs).to(self.device).float() if isinstance(obs, np.ndarray) else obs.to(self.device).float()
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        with torch.no_grad():
            q = self.q_net(obs_t)

        greedy = q.argmax(dim=1).cpu().numpy().astype(int)
        if epsilon > 0.0:
            B = greedy.shape[0]
            mask = np.random.rand(B) < float(epsilon)
            if mask.any():
                greedy[mask] = np.random.randint(0, self.env.action_space.n, size=mask.sum())
        return int(greedy[0]) if greedy.shape[0] == 1 else greedy

    # ---------- training loop (no RND warmup, no intrinsic) ----------
    def train(self, total_timesteps, max_episode_steps, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.99999671):
        current_timestep = 0
        episode_return = 0.0
        episode_step = 0
        episode_num = 0

        obs_raw, _ = self.env.reset()
        obs = self.process_obs(obs_raw)
        self.increment_visit_counts()

        epsilon = float(epsilon_start)

        stats = EpisodeStats([], [], [])

        while current_timestep < total_timesteps:
            action = self.act(obs, epsilon)

            next_obs_raw, ext_reward, terminated, truncated, _ = self.env.step(action)
            next_obs = self.process_obs(next_obs_raw)
            self.increment_visit_counts()

            done = bool(terminated or truncated)
            term = bool(terminated)
            timeout = bool(truncated)

            self.replay_buffer.add(
                obs=obs,
                act=action,
                ext_rew=self.extrinsic_coef * float(ext_reward),
                term=term,
                timeout=timeout,
                next_obs=next_obs,
            )

            # Learn
            if self.replay_buffer.get_stored_size() >= self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                self.update_dqn(batch)


            # Log coarse signals
            if current_timestep % 1000 == 0 and current_timestep > 0:
                wandb.log({
                    "a/ext_reward": ext_reward,
                    "a/epsilon": epsilon,
                }, step=current_timestep)

            obs = next_obs
            episode_return += self.extrinsic_coef * float(ext_reward)
            episode_step += 1
            current_timestep += 1

            # target sync
            if current_timestep % self.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            # episode end
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

                if epsilon > epsilon_end:
                    epsilon *= epsilon_decay

                obs_raw, _ = self.env.reset()
                obs = self.process_obs(obs_raw)
                self.increment_visit_counts()
                episode_return = 0.0
                episode_step = 0
                episode_num += 1

            # periodic validation
            if current_timestep % 10000 == 0 and current_timestep > 0:
                evaluate_dqn(self, self.eval_env, current_timestep, save_dir="eval_mrl")


    def update_dqn(self, batch):
        obs = torch.from_numpy(batch["obs"]).float().to(self.device)                 # (B, D)
        next_obs = torch.from_numpy(batch["next_obs"]).float().to(self.device)       # (B, D)
        act = torch.from_numpy(batch["act"].squeeze(-1)).long().to(self.device)      # (B,)
        ext = torch.from_numpy(batch["ext_rew"].squeeze(-1)).float().to(self.device)     # (B,)
        term = torch.from_numpy(batch["term"].squeeze(-1)).float().to(self.device)   # (B,)
        timeout = torch.from_numpy(batch["timeout"].squeeze(-1)).float().to(self.device)

        q_s_online = self.q_net(obs)                         # (B, A)
        q_sa = q_s_online.gather(1, act.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            v_s = q_s_online.max(1, keepdim=True)[0]
            logsum_s = torch.logsumexp((q_s_online - v_s) / self.tau, dim=1, keepdim=True)
            log_pi_s = q_s_online - v_s - self.tau * logsum_s                 # (B, A)
            log_pi_sa = log_pi_s.gather(1, act.unsqueeze(1)).squeeze(1)       # (B,)
            log_pi_sa = torch.clamp(log_pi_sa, min=self.lo, max=0.0)
        munchausen_reward = ext + self.alpha_m * log_pi_sa                    # (B,)

        with torch.no_grad():
            q_sp_online = self.q_net(next_obs)                                # (B, A)
            v_sp_on = q_sp_online.max(1, keepdim=True)[0]
            logsum_sp = torch.logsumexp((q_sp_online - v_sp_on) / self.tau, dim=1, keepdim=True)
            log_pi_sp = q_sp_online - v_sp_on - self.tau * logsum_sp          # (B, A)
            pi_sp = F.softmax(q_sp_online / self.tau, dim=1)                  # (B, A)

            q_sp_target = self.target_q_net(next_obs)                         # (B, A)
            soft_backup = (pi_sp * (q_sp_target - self.tau * log_pi_sp)).sum(dim=1)  # (B,)

            bootstrap_mask = 1.0 - term

            target = munchausen_reward + bootstrap_mask * self.gamma * soft_backup   # (B,)

        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            gap = (q_s_online.max(1, keepdim=True)[0] - q_s_online).mean().item()
        wandb.log({"loss/td": loss.item(), "gap/full_mean_gap": gap}, commit=False)



def main():
    wandb.init(project="dqn", name="mrl")

    ENV_NAME = "Fixed-DoorKey-v0"
    max_episode_steps = 400
    total_timesteps = 1_000_000

    env = customised_doorkey.make_fixed_doorkey_env(
        size=10,
        key_color="red", key_pos=(1, 8),     
        door_color="red", door_pos=(5, 5),   
        goal_pos=(8, 1),
        agent_start_pos=(1, 1), agent_start_dir=0,
        wall_cells=[(7, 2), (8, 2)],
        # extra_keys=[((9, 1), "blue")],
        # extra_doors=[((12, 7), "blue", True)],
        ensure_door_in_wall=True,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )

    eval_env = customised_doorkey.make_fixed_doorkey_env(
        size=10,
        key_color="red", key_pos=(1, 8),     
        door_color="red", door_pos=(5, 5),   
        goal_pos=(8, 1),
        agent_start_pos=(1, 1), agent_start_dir=0,
        wall_cells=[(7, 2), (8, 2)],
        # extra_keys=[((9, 1), "blue")],
        # extra_doors=[((12, 7), "blue", True)],
        ensure_door_in_wall=True,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )

    # ----- configs -----
    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": 512,
        "lr": 2.5e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "replay_buffer_size": 1_000_000,
        "target_update_freq": 5000,
        "learning_starts": 20_000,
        "max_grad_norm": 10.0
    })

    # Munchausen hyperparams
    mrl_cfg = type("MRLConfig", (), {
        "tau": 0.06,          # entropy temperature
        "alpha_m": 0.3,       # Munchausen coefficient
        "lo": -1.0,           # clip lower bound for log π
        "extrinsic_coef": 1.0 
    })

    agent = DQN_MRL_Agent(
        env=env,
        env_name=ENV_NAME,
        eval_env=eval_env,
        dqn_cfg=dqn_cfg,
        mrl_cfg=mrl_cfg
    )

    start = time.time()
    agent.train(
        total_timesteps=total_timesteps,
        max_episode_steps=max_episode_steps,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9997
    )
    end = time.time()
    print(f"Training finished in {(end - start) / 60:.2f} minutes.")
    agent.print_visit_counts()


if __name__ == "__main__":
    main()
