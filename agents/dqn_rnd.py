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

        self.eps_start, self.eps_end, self.decay_steps = 1.0, 0.05, 200_000

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

        stats = EpisodeStats([], [], [])
        epsilon = self.rnd_cfg.epsilon_start

        self._fit_obs_rms_warmup()

        obs_raw, _ = self.env.reset()
        obs = self.process_obs(obs_raw)
        self.update_obs_rms(obs)
        self.increment_visit_counts()

        while current_timestep < total_timesteps:
            frac = min(1.0, current_timestep / self.decay_steps)
            epsilon = self.eps_start + (self.eps_end - self.eps_start) * frac

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
            # self.update_obs_rms(next_obs_tensor.cpu().numpy().squeeze(0))
            int_reward_tensor = self.compute_intrinsic_reward(next_obs_tensor)
            int_reward = float(int_reward_tensor.item())

            discounted_r = self.reward_filter.update(int_reward)
            self.reward_rms.update(np.array([discounted_r]))
            norm_int_reward = int_reward / np.sqrt(np.maximum(self.reward_rms.var, 1e-8))

            # if current_timestep % 1000 == 0:
            #     agent_x, agent_y = map(int, self.env.unwrapped.agent_pos)
            #     wandb.log({f"int_reward/cell({agent_y - 1},{agent_x - 1})": norm_int_reward}, step=current_timestep)

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

                # batch = self.replay_buffer.sample(self.batch_size)
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

                # if epsilon > self.rnd_cfg.epsilon_end:
                #     epsilon *= self.rnd_cfg.epsilon_decay

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
                plot_rnd_intrinsic_three_panels_agg(self, agg="mean", step=current_timestep)
                

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


def plot_rnd_intrinsic_three_panels_agg(agent, agg="max", normalized=True, step=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from minigrid.core.world_object import Door, Key

    def _apply_observation_wrappers(top_env, raw_obs):
        """Replay observation() transforms of wrapper stack (bottom-up)."""
        wrappers, env = [], top_env
        while hasattr(env, "env"):
            if hasattr(env, "observation"):
                wrappers.append(env)
            env = env.env
        for w in reversed(wrappers):  # innermost first
            raw_obs = w.observation(raw_obs)
        return raw_obs

    def _obs_vec_at_pose(x, y, d):
        base.agent_pos = (x + 1, y + 1)
        base.agent_dir = d
        raw = base.gen_obs()
        wrapped = _apply_observation_wrappers(agent.eval_env, raw)
        return agent.process_obs(wrapped, env_for_pose=base)

    def _find_primary_door(b):
        """Pick a single 'primary' door to manipulate."""
        H, W = b.grid.height, b.grid.width
        primary = None
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                obj = b.grid.get(x, y)
                if isinstance(obj, Door):
                    # Heuristic: prefer locked red/yellow door; else first door.
                    if primary is None:
                        primary = (x, y, obj)
                    # Prefer locked main door if available
                    if getattr(obj, "is_locked", False):
                        return (x, y, obj)
        return primary  # may be None if no doors exist

    def _remove_keys_of_color(b, color: str):
        """Remove keys of a given color from the grid (simulate 'picked up')."""
        H, W = b.grid.height, b.grid.width
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                obj = b.grid.get(x, y)
                if isinstance(obj, Key) and getattr(obj, "color", None) == color:
                    b.grid.set(x, y, None)


    base = agent.eval_env.unwrapped
    H_in, W_in = base.grid.height - 2, base.grid.width - 2  # interior only

    # normalization 
    use_norm = normalized and hasattr(agent, "reward_rms") and hasattr(agent.reward_rms, "var")
    denom = None
    if use_norm:
        try:
            denom = float(np.sqrt(max(float(agent.reward_rms.var), 1e-8)))
        except Exception:
            use_norm, denom = False, None

    expected_dim = None
    if hasattr(agent, "obs_mean"):
        try:
            expected_dim = int(agent.obs_mean.shape[-1])
        except Exception:
            expected_dim = None


    configs = [
        ("Closed, no key", dict(door_open=False, has_key=False)),
        ("Closed, has key", dict(door_open=False, has_key=True)),
        ("Open, has key",   dict(door_open=True,  has_key=True)),
    ]

    maps, vmin, vmax = [], +1e9, -1e9

    for _, cfg in configs:
        agent.eval_env.reset()
        base = agent.eval_env.unwrapped

        # Detect primary door (pos + color)
        pd = _find_primary_door(base)
        pd_pos, pd_color = None, None
        if pd is not None:
            x_d, y_d, door_obj = pd
            pd_pos = (x_d, y_d)
            pd_color = getattr(door_obj, "color", None)

        # Equip key if requested (use detected door color if available)
        base.carrying = None
        if cfg["has_key"]:
            key_color = pd_color if pd_color is not None else "yellow"
            base.carrying = Key(key_color)
            # Remove keys of that color from the grid to reflect "already picked up"
            _remove_keys_of_color(base, key_color)

        # Toggle only the primary door if it exists; leave other doors as-is
        if pd_pos is not None:
            x_d, y_d = pd_pos
            obj = base.grid.get(x_d, y_d)
            if isinstance(obj, Door):
                obj.is_open = bool(cfg["door_open"])
                # If closed, keep it locked (typical DoorKey semantics); if open, unlock
                obj.is_locked = not obj.is_open
                base.grid.set(x_d, y_d, obj)

        # Sweep positions/directions
        M_dir = np.empty((4, H_in, W_in), dtype=np.float32)
        for d in range(4):
            for yy in range(H_in):
                for xx in range(W_in):
                    obs_vec = _obs_vec_at_pose(xx, yy, d)
                    obs_t = agent.obs_to_float_tensor(obs_vec)
                    if obs_t.ndim == 1:
                        obs_t = obs_t.unsqueeze(0)

                    if expected_dim is not None and obs_t.shape[-1] != expected_dim:
                        raise ValueError(
                            f"Observation dim mismatch during plotting: got {obs_t.shape[-1]}, "
                            f"expected {expected_dim}. Ensure wrappers/process_obs match training."
                        )

                    with torch.no_grad():
                        bonus = float(agent.compute_intrinsic_reward(obs_t).item())
                    if use_norm and denom is not None:
                        bonus /= denom
                    M_dir[d, yy, xx] = bonus

        M = np.nanmax(M_dir, axis=0) if agg == "max" else np.nanmean(M_dir, axis=0)
        maps.append(M)
        if np.isfinite(M).any():
            vmin = min(vmin, float(np.nanmin(M)))
            vmax = max(vmax, float(np.nanmax(M)))

    fig, axs = plt.subplots(1, 4, figsize=(28, 9), dpi=200,
                            gridspec_kw={"width_ratios": [1, 1, 1, 0.04]})
    cax = axs[3]
    counts = getattr(agent, "visit_counts", None)

    def _safe_matrix(M):
        return np.zeros_like(M, dtype=np.float32) if not np.isfinite(M).any() else M

    last_im = None
    for ax, (title, _), M in zip(axs[:3], configs, maps):
        Mplot = _safe_matrix(M)
        last_im = ax.imshow(
            Mplot, cmap="viridis",
            vmin=None if not np.isfinite(vmin) else vmin,
            vmax=None if not np.isfinite(vmax) else vmax,
            origin="upper", interpolation="nearest"
        )
        suffix = " (norm)" if use_norm else ""
        ax.set_title(f"{title} ({agg} over dir){suffix}", fontsize=12, pad=6)
        ax.set_xticks(range(W_in)); ax.set_yticks(range(H_in))

        if isinstance(counts, np.ndarray) and counts.shape == (H_in, W_in):
            for yy in range(H_in):
                for xx in range(W_in):
                    ax.text(xx, yy, str(int(counts[yy, xx])),
                            ha="center", va="center", fontsize=8, color="white")

    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("RND intrinsic" + (" (normalized)" if use_norm else " (raw)"), labelpad=6)

    try:
        import wandb
        if step is not None:
            wandb.log(
                {f"heatmap/rnd_intrinsic_three_panels_{agg}{'_norm' if use_norm else ''}": wandb.Image(fig)},
                step=step
            )
    except Exception:
        pass

    plt.close(fig)




def main():
    wandb.init(project="dqn", name="rnd")
    ENV_NAME = "Fixed-DoorKey-v0"

    max_episode_steps = 1600
    total_timesteps = 1_000_000

    # env = customised_doorkey.make_fixed_doorkey_env(
    #     size=10,
    #     key_color="red", key_pos=(1, 8),     
    #     door_color="red", door_pos=(5, 5),   
    #     goal_pos=(8, 1),
    #     agent_start_pos=(1, 1), agent_start_dir=0,
    #     wall_cells=[(7, 2), (8, 2)],
    #     # extra_keys=[((9, 1), "blue")],
    #     # extra_doors=[((12, 7), "blue", True)],
    #     ensure_door_in_wall=True,
    #     render_mode="rgb_array",
    #     max_episode_steps=max_episode_steps,
    # )

    # eval_env = customised_doorkey.make_fixed_doorkey_env(
    #     size=10,
    #     key_color="red", key_pos=(1, 8),     
    #     door_color="red", door_pos=(5, 5),   
    #     goal_pos=(8, 1),
    #     agent_start_pos=(1, 1), agent_start_dir=0,
    #     wall_cells=[(7, 2), (8, 2)],
    #     # extra_keys=[((9, 1), "blue")],
    #     # extra_doors=[((12, 7), "blue", True)],
    #     ensure_door_in_wall=True,
    #     render_mode="rgb_array",
    #     max_episode_steps=max_episode_steps,
    # )
    

    env = customised_doorkey.make_fixed_doorkey_env(
        size=16,
        key_color="blue", key_pos=(9, 1),     
        door_color="blue", door_pos=(12, 7),   
        goal_pos=(9, 14),
        agent_start_pos=(1, 1), agent_start_dir=0,
        wall_cells=[(9, 7), (10, 7), (11, 7), (13, 7), (14, 7)],
        # extra_keys=[((9, 1), "blue")],
        # extra_doors=[((12, 7), "blue", True)],
        ensure_door_in_wall=True,
        empty_cells=[(8, 5)],
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )

    eval_env = customised_doorkey.make_fixed_doorkey_env(
        size=16,
        key_color="blue", key_pos=(9, 1),     
        door_color="blue", door_pos=(12, 7),   
        goal_pos=(9, 14),
        agent_start_pos=(1, 1), agent_start_dir=0,
        wall_cells=[(9, 7), (10, 7), (11, 7), (13, 7), (14, 7)],
        # extra_keys=[((9, 1), "blue")],
        # extra_doors=[((12, 7), "blue", True)],
        ensure_door_in_wall=True,
        empty_cells=[(8, 5)],
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )

    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": 1024, #512,
        "lr": 2.5e-4, #1e-4
        "gamma": 0.99, #0.99
        "batch_size": 128, #128
        "replay_buffer_size": 500_000, #1_000_000,
        "target_update_freq": 2000 #2000
    })

    rnd_cfg = type("RNDConfig", (), {
        "intrinsic_coef": 1.0, #1.0
        "extrinsic_coef": 2.0, #2.0
        "lr": 1e-4, #1e-4,
        "learning_starts": 10_000, #1000
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.999, #0.9998
        "rnd_mask_prob": 0.5 #0.25
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
