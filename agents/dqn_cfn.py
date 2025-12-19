import os
import sys
import time
import math
import logging
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import wandb
from matplotlib import pyplot as plt
from cpprb import ReplayBuffer

from CFN.CFN import CoinFlipNetwork
from CFN.cfn_buffer import CFNReplayBufferWrapper
from CFN.priority_util import get_coin_flips
import customised_doorkey
from networks.encoder import DDQN  

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
log = logging.getLogger(__name__)


class DQN_CFNAgent:
    def __init__(self, env, eval_env, env_name, dqn_cfg, cfn_cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.eval_env = eval_env
        self.env_name = env_name

        # observation shape
        obs_space = self.env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            H_img, W_img, _ = obs_space["image"].shape
        else:
            H_img, W_img, _ = obs_space.shape

        # MiniGrid vocab sizes
        self.N_TYPE = 11 # object types in MiniGrid (wall, key etc)
        self.N_COLOR = 6 # object color
        self.N_STATE = 4 # open closed ocked unused 

        # interior dims (without outer walls)
        H_int = self.env.unwrapped.grid.height - 2
        W_int = self.env.unwrapped.grid.width - 2
        self.H_int, self.W_int = H_int, W_int

        # one-hot caches
        self._eye_type = np.eye(self.N_TYPE, dtype=np.float32)
        self._eye_color = np.eye(self.N_COLOR, dtype=np.float32)
        self._eye_state = np.eye(self.N_STATE, dtype=np.float32)

        # Q-state dims: TYPE one-hot per cell + pos one-hot + dir one-hot + three flags
        self.type_onehot_size = H_img * W_img * self.N_TYPE          
        self.pos_onehot_size = H_img * W_img                         # img so that it aligns with N_TYPE
        self.dir_size = 4
        self.flags_size = 3  # has_key, door_open, key_match
        self.state_dim = self.type_onehot_size + self.pos_onehot_size + self.dir_size + self.flags_size

        # CFN state: compact semantic vector (interior pos one-hot + dir + has_key + door_open + key_match)
        # interior pos one-hot = H_int * W_int
        self.cfn_pos_dim = H_int * W_int
        self.cfn_state_dim = self.cfn_pos_dim + 4 + 1 + 1 + 1  # dir onehot + 3 flags (will encode flags as 0/1 scalars)
        
        # action dim
        self.act_dim = self.env.action_space.n

        # networks
        self.q_net = DDQN(self.state_dim, self.act_dim, dqn_cfg.hidden_size, is_cnn=False).to(self.device)
        self.target_q_net = DDQN(self.state_dim, self.act_dim, dqn_cfg.hidden_size, is_cnn=False).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=dqn_cfg.lr, eps=1.5e-4)

        # replay buffer
        self.replay_buffer = ReplayBuffer(
            dqn_cfg.replay_buffer_size,
            env_dict={
                "obs": {"shape": (self.state_dim,), "dtype": np.float32},
                "act": {"shape": 1, "dtype": np.int16},
                "rew": {"shape": 1, "dtype": np.float32},
                "intr": {"shape": 1, "dtype": np.float32},
                "term": {"shape": 1, "dtype": np.bool_},
                "timeout": {"shape": 1, "dtype": np.bool_},
                "next_obs": {"shape": (self.state_dim,), "dtype": np.float32},
            }
        )

        # CFN
        self.coin_flip_dim = cfn_cfg.cfn_coin_flip_dim
        self.cfn = CoinFlipNetwork(state_dim=self.cfn_state_dim, coin_dim=self.coin_flip_dim, device=self.device).to(self.device)
        self.cfn_optimizer = optim.Adam(self.cfn.net.parameters(), lr=cfn_cfg.cfn_lr, betas=(0.9, 0.999), eps=1e-8)
        self.cfn_buffer = CFNReplayBufferWrapper(size=cfn_cfg.cfn_replay_buffer_size, obs_shape=(self.cfn_state_dim,), coin_flip_dim=self.coin_flip_dim, alpha=0.5)

        # hyperparams
        self.gamma = dqn_cfg.gamma
        self.batch_size = dqn_cfg.batch_size
        self.target_update_freq = dqn_cfg.target_update_freq
        self.learning_starts = dqn_cfg.learning_starts

        # lambda schedule for intrinsic
        self.lambda_init = cfn_cfg.cfn_intrinsic_scale
        self.lambda_bonus = self.lambda_init
        self.lambda_decay_start = 400_000
        self.lambda_decay_end = 600_000

        # epsilon
        self.eps_start = 1.0
        self.eps_end = 0.02
        self.eps_decay_steps = 200_000
        self.eval_epsilon = 0.001
        self.epsilon = self.eps_start

        self.cfn_batch_size = cfn_cfg.cfn_batch_size

        # counters
        self.step_count = 0
        self.visit_counts = np.zeros((H_int, W_int), dtype=np.int32)

    # ---------- helpers for environment state ----------
    def _door_ahead(self, base):
        # checks if door is the next object in front of the agent
        from minigrid.core.world_object import Door
        b = base
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dx, dy = dirs[int(getattr(b, "agent_dir", 0)) % 4]
        x, y = map(int, b.agent_pos)
        obj = b.grid.get(x + dx, y + dy)
        return obj if isinstance(obj, Door) else None

    def _has_key(self, base) -> int:
        # checks if the agent is carrying a key
        from minigrid.core.world_object import Key
        c = getattr(base, "carrying", None)
        return int(isinstance(c, Key))

    def _door_ahead_open(self, base) -> int:
        # checks if the door ahead is open
        d = self._door_ahead(base)
        return int(d is not None and d.is_open)

    def _key_matches_door_ahead(self, base) -> int:
        # checks if the key carried matches the door ahead
        d = self._door_ahead(base)
        c = getattr(base, "carrying", None)
        if d is None or c is None:
            return 0
        return int(getattr(c, "color", None) == getattr(d, "color", None))

    # ---------- CFN ----------
    def cfn_obs(self, base=None) -> np.ndarray:
        b = self.env.unwrapped if base is None else base
        H_int, W_int = self.H_int, self.W_int
        x, y = map(int, b.agent_pos)
        xi, yi = x - 1, y - 1 # bc cfn uses interior coords

        pos = np.zeros(self.cfn_pos_dim, dtype=np.float32)
        if 0 <= xi < W_int and 0 <= yi < H_int:
            pos[yi * W_int + xi] = 1.0

        d = int(getattr(b, "agent_dir", 0)) % 4
        dir_oh = np.zeros(4, dtype=np.float32)
        dir_oh[d] = 1.0

        hk = float(self._has_key(b))
        da = float(self._door_ahead_open(b))
        km = float(self._key_matches_door_ahead(b))

        return np.concatenate([pos, dir_oh, np.array([hk, da, km], dtype=np.float32)], axis=0)

    # ---------- Q-network observation (one-hot TYPE + pos one-hot + dir + flags) ----------
    def process_obs(self, obs_raw, env_for_pose=None):
        img = np.asarray(obs_raw["image"], dtype=np.int16)  # H, W, 3
        H, W = img.shape[:2]

        # TYPE channel -> ints 0..N_TYPE-1
        type_grid = img[..., 0].clip(0, self.N_TYPE - 1).astype(np.int32)  # (H,W)

        # Convert to one-hot using cached identity: result shape (H, W, N_TYPE)
        onehot = self._eye_type[type_grid]  # dtype float32, shape (H, W, N_TYPE)
        type_onehot_flat = onehot.reshape(-1)  # length H*W*N_TYPE

        # position one-hot over full image (H x W) — place 1.0 at agent location
        env0 = env_for_pose if env_for_pose is not None else self.env
        ax, ay = map(int, env0.unwrapped.agent_pos)
        pos_onehot = np.zeros((H, W), dtype=np.float32)
        if 0 <= ay < H and 0 <= ax < W:
            pos_onehot[ay, ax] = 1.0
        pos_onehot_flat = pos_onehot.reshape(-1)

        # dir one-hot
        d = int(getattr(env0.unwrapped, "agent_dir", 0)) % 4
        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_onehot[d] = 1.0

        # flags
        has_key_f = float(self._has_key(env0.unwrapped))
        door_open_f = float(self._door_ahead_open(env0.unwrapped))
        key_match_f = float(self._key_matches_door_ahead(env0.unwrapped))

        # concatenate
        return np.concatenate([
            type_onehot_flat.astype(np.float32, copy=False),
            pos_onehot_flat,
            dir_onehot,
            np.array([has_key_f, door_open_f, key_match_f], dtype=np.float32)
        ], axis=0)

    # ---------- CFN bonus calculation and buffer add ----------
    @torch.no_grad()
    def cfn_bonus_and_buffer(self, base_env=None):
        phi = self.cfn_obs(base_env)  # compact semantic vector
        obs_t = torch.from_numpy(phi).unsqueeze(0).to(self.device).float()

        # compute squared output norm and raw bonus
        sq = self.cfn.compute_squared_output_norm(obs_t, update_prior_stats=False)
        bonus = torch.sqrt(sq / float(self.coin_flip_dim))
        # update prior stats
        prior_out = self.cfn.prior(obs_t)
        self.cfn._update_prior_stats(prior_out)

        # sample coin flips and add to CFN buffer
        coin_flip = get_coin_flips(self.coin_flip_dim)
        self.cfn_buffer.add(
            obs=phi.astype(np.float32, copy=False),
            coin_flip=coin_flip.detach().cpu().numpy().astype(np.float32, copy=False),
            priority=1.0
        )
        return float(bonus.item())

    def _update_lambda(self, step: int):
        if step < self.lambda_decay_start:
            self.lambda_bonus = self.lambda_init
        elif step >= self.lambda_decay_end:
            self.lambda_bonus = 0.0
        else:
            frac = (step - self.lambda_decay_start) / (self.lambda_decay_end - self.lambda_decay_start)
            self.lambda_bonus = float(self.lambda_init * (1.0 - frac))

    def _update_visit_counts(self, base_env=None):
        b = base_env.unwrapped if base_env is not None else self.env.unwrapped
        x, y = map(int, b.agent_pos)
        xi, yi = x - 1, y - 1
        if 0 <= xi < self.W_int and 0 <= yi < self.H_int:
            self.visit_counts[yi, xi] += 1

    def act(self, obs_tensor, epsilon: float):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            q = self.q_net(obs_tensor)
            return int(q.argmax(dim=1).item())

    # ---------- training loop ----------
    def train(self, total_timesteps, max_episode_steps):
        current_timestep = 0
        episode_return_aug = 0.0
        episode_step = 0
        episode_num = 0

        obs_raw, _ = self.env.reset()
        obs = self.process_obs(obs_raw)
        self._update_visit_counts()

        while current_timestep < total_timesteps:
            # epsilon schedule
            if current_timestep < self.eps_decay_steps:
                frac = current_timestep / max(1, self.eps_decay_steps)
                self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)
            else:
                self.epsilon = self.eps_end

            # lambda schedule
            self._update_lambda(current_timestep)

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.act(obs_tensor, self.epsilon)

            next_obs_raw, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = self.process_obs(next_obs_raw)
            done = bool(terminated or truncated)

            # CFN intrinsic + buffer add
            bonus_raw = self.cfn_bonus_and_buffer()

            # store transition (cast to float32)
            self.replay_buffer.add(
                obs=obs.astype(np.float32),
                act=int(action),
                rew=float(reward),
                intr=float(bonus_raw),
                term=bool(terminated),
                timeout=bool(truncated),
                next_obs=next_obs.astype(np.float32),
            )

            # Q update
            if self.replay_buffer.get_stored_size() >= max(self.batch_size, self.learning_starts):
                batch = self.replay_buffer.sample(self.batch_size)
                self.update_q(batch, current_timestep)

            # CFN update
            if self.cfn_buffer.get_stored_size() >= self.cfn_batch_size:
                obs_bc, coin_bc, idx = self.cfn_buffer.sample_with_indices(self.cfn_batch_size)
                self.update_cfn(obs_bc, coin_bc, current_timestep)
                self.cfn_buffer.update_priorities(idx, obs_bc, self.cfn, self.coin_flip_dim)

            # target update
            if current_timestep > 0 and current_timestep % self.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            # bookkeeping
            obs = next_obs
            episode_return_aug += float(reward + self.lambda_bonus * bonus_raw)
            episode_step += 1
            current_timestep += 1
            self.step_count += 1
            self._update_visit_counts()

            # episodic end
            if done or episode_step >= max_episode_steps:
                # scalar wandb logging only
                wandb.log({
                    "charts/episodic_return": episode_return_aug,
                    "charts/episodic_length": episode_step,
                    "charts/episode_num": episode_num,
                    "training/lambda": self.lambda_bonus,
                    "training/epsilon": self.epsilon,
                }, step=current_timestep)

                log.info(f"Episode {episode_num} | Steps: {episode_step} | AugReturn: {episode_return_aug:.2f} | ε: {self.epsilon:.3f} | λ: {self.lambda_bonus:.4f} | T: {current_timestep}")

                obs_raw, _ = self.env.reset()
                obs = self.process_obs(obs_raw)
                episode_return_aug = 0.0
                episode_step = 0
                episode_num += 1

            # periodic eval (numerical only)
            if current_timestep % 10000 == 0 and current_timestep > 0:
                evaluate_dqn(self, self.eval_env, current_timestep, save_dir="checkpoints_cfn", epsilon_eval=0.0)

    # ---------- Q / CFN update functions ----------
    def update_q(self, batch, step):
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        act = torch.from_numpy(batch["act"].squeeze(-1)).long().to(self.device)
        rew = torch.from_numpy(batch["rew"].squeeze(-1)).float().to(self.device)
        intr = torch.from_numpy(batch["intr"].squeeze(-1)).float().to(self.device)
        term = torch.from_numpy(batch["term"].squeeze(-1)).bool().to(self.device)
        timeout = torch.from_numpy(batch["timeout"].squeeze(-1)).bool().to(self.device)

        done = term | timeout
        bootstrap_mask = (~done).float()
        aug_rew = rew + self.lambda_bonus * intr

        self.optimizer.zero_grad(set_to_none=True)

        q_vals = self.q_net(obs)
        q_val = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_obs).argmax(1)
            next_q_vals_target = self.target_q_net(next_obs)
            max_next_q = next_q_vals_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = aug_rew + bootstrap_mask * self.gamma * max_next_q

        loss = F.smooth_l1_loss(q_val, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # occasional logging
        if step % 1000 == 0:
            with torch.no_grad():
                wandb.log({
                    "training/dqn_loss": float(loss.item()),
                    "training/q_values_mean": float(q_val.mean().item()),
                    "training/target_mean": float(target.mean().item()),
                }, step=step)

    def update_cfn(self, obs_batch, coin_flip_batch, step):
        if isinstance(obs_batch, np.ndarray):
            obs_t = torch.from_numpy(obs_batch).float().to(self.device)
        else:
            obs_t = obs_batch.float().to(self.device)

        if isinstance(coin_flip_batch, np.ndarray):
            coins = torch.from_numpy(coin_flip_batch).float().to(self.device)
        else:
            coins = coin_flip_batch.float().to(self.device)

        self.cfn_optimizer.zero_grad(set_to_none=True)
        pred = self.cfn(obs_t, update_prior_stats=False)
        loss = F.mse_loss(pred, coins)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.cfn.net.parameters(), max_norm=10.0)
        self.cfn_optimizer.step()

        if step % 5000 == 0:
            wandb.log({"training/cfn_loss": float(loss.item())}, step=step)


# ---------- evaluation (numerical only) ----------
def evaluate_dqn(agent, eval_env, step, save_dir="eval-flat", num_episodes=5, epsilon_eval=None):
    os.makedirs(save_dir, exist_ok=True)
    returns, lengths = [], []
    if epsilon_eval is None:
        epsilon_eval = agent.eval_epsilon

    was_training = agent.q_net.training
    agent.q_net.eval()
    pos_history = []

    with torch.no_grad():
        for ep in range(num_episodes):
            obs_raw, _ = eval_env.reset()
            obs = agent.process_obs(obs_raw, env_for_pose=eval_env)
            done = False
            total_return = 0.0
            ep_len = 0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                action = agent.act(obs_t, epsilon=epsilon_eval)
                next_obs_raw, reward, terminated, truncated, _ = eval_env.step(action)
                done = bool(terminated or truncated)
                obs = agent.process_obs(next_obs_raw, env_for_pose=eval_env)
                base = eval_env.unwrapped
                pos_history.append((tuple(map(int, base.agent_pos)), int(getattr(base, "agent_dir", 0)), action))
                total_return += float(reward)
                ep_len += 1
            returns.append(total_return)
            lengths.append(ep_len)

    if was_training:
        agent.q_net.train()

    returns = np.array(returns, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.int32)
    np.savez(os.path.join(save_dir, f"eval_step_{step}.npz"), returns=returns, lengths=lengths)

    wandb.log({
        "eval/mean_return": float(returns.mean()),
        "eval/std_return": float(returns.std()),
        "eval/mean_length": float(lengths.mean()),
    }, step=step)

    print(f"[EVAL] step {step} | avg return: {returns.mean():.3f} | avg len: {lengths.mean():.1f}")
    print("EVAL unique positions:", len(set(p for p, _, _ in pos_history)))
    return returns.mean()


# ---------- main ----------
def main():
    wandb.init(project="dqn", name="cfn-patched", reinit=True)

    total_timesteps = 1_000_000
    max_episode_steps = 1600

    env = customised_doorkey.make_fixed_doorkey_env(
        size=16,
        key_color="blue", key_pos=(9, 1),
        door_color="blue", door_pos=(12, 7),
        goal_pos=(9, 14),
        agent_start_pos=(1, 1), agent_start_dir=0,
        wall_cells=[(9, 7), (10, 7), (11, 7), (13, 7), (14, 7)],
        ensure_door_in_wall=True,
        empty_cells=[(8, 5)],
        render_mode=None,  # NO render during training
        max_episode_steps=max_episode_steps,
    )
    eval_env = customised_doorkey.make_fixed_doorkey_env(
        size=16,
        key_color="blue", key_pos=(9, 1),
        door_color="blue", door_pos=(12, 7),
        goal_pos=(9, 14),
        agent_start_pos=(1, 1), agent_start_dir=0,
        wall_cells=[(9, 7), (10, 7), (11, 7), (13, 7), (14, 7)],
        ensure_door_in_wall=True,
        empty_cells=[(8, 5)],
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )

    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": 512, "lr": 3e-4, "gamma": 0.99,
        "batch_size": 256, "replay_buffer_size": 500_000,
        "target_update_freq": 1000, "learning_starts": 10_000
    })()

    cfn_cfg = type("CFNConfig", (), {
        "cfn_coin_flip_dim": 20,
        "cfn_lr": 1e-4,
        "cfn_replay_buffer_size": 500_000,
        "cfn_batch_size": 1024,
        "cfn_intrinsic_scale": 0.05,
    })()

    wandb.config.update({"dqn": dqn_cfg.__dict__})
    wandb.config.update({"cfn": cfn_cfg.__dict__})

    agent = DQN_CFNAgent(env=env, eval_env=eval_env, env_name="Fixed-DoorKey-v0", dqn_cfg=dqn_cfg, cfn_cfg=cfn_cfg)
    start = time.time()
    agent.train(total_timesteps=total_timesteps, max_episode_steps=max_episode_steps)
    end = time.time()
    print(f"Training finished in {(end - start) / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
