import os, sys, time, math, logging
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import wandb
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
from cpprb import ReplayBuffer
from minigrid.wrappers import FullyObsWrapper
from minigrid.core.world_object import Door, Key
from networks.ddqn import DDQN
from CFN.CFN import CoinFlipNetwork
from CFN.cfn_buffer import CFNReplayBufferWrapper
from CFN.priority_util import get_coin_flips
import customised_doorkey

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
log = logging.getLogger(__name__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
class DQN_CFNAgent:
    def __init__(self, env, eval_env, env_name, dqn_cfg, cfn_cfg):
        # device + AMP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = torch.cuda.is_available()
        self.scaler_q = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.scaler_cfn = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        if self.use_amp:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # env refs
        self.env, self.eval_env, self.env_name = env, eval_env, env_name

        # obs/action sizes
        obs_space = self.env.observation_space
        self.obs_shape_hw_c = obs_space["image"].shape if isinstance(obs_space, gym.spaces.Dict) else obs_space.shape
        self.obs_shape = (self.obs_shape_hw_c[2], self.obs_shape_hw_c[0], self.obs_shape_hw_c[1])
        self.act_dim = self.env.action_space.n

        # grid sizes
        H, W, _ = self.env.observation_space["image"].shape
        base = self.env.unwrapped
        H_int, W_int = base.grid.height - 2, base.grid.width - 2

        # MiniGrid vocab
        self.N_TYPE, self.N_COLOR, self.N_STATE = 11, 6, 4
        self.FEAT_PER_CELL = self.N_TYPE + self.N_COLOR + self.N_STATE

        # CFN one-hot dims (pos + dir + 3 binary flags)
        self._cfn_H_in, self._cfn_W_in = H_int, W_int
        self._cfn_pos_dim = H_int * W_int
        self._cfn_dir_dim = 4
        self._cfn_flag_dim = 2
        self.cfn_state_dim = self._cfn_pos_dim + self._cfn_dir_dim + 3 * self._cfn_flag_dim

        # one-hot caches
        self._eye_type = np.eye(self.N_TYPE, dtype=np.float32)
        self._eye_color = np.eye(self.N_COLOR, dtype=np.float32)
        self._eye_state = np.eye(self.N_STATE, dtype=np.float32)
        self._agent_eye = np.eye(H * W, dtype=np.uint8).reshape(H * W, H, W)

        # flat DQN state = cells + (ax,ay) + dir(4)
        self.state_dim = H * W * self.FEAT_PER_CELL + 2 + 4

        # Q nets + opt
        self.q_net = DDQN(self.state_dim, self.act_dim, dqn_cfg.hidden_size, is_cnn=False).to(self.device)
        self.target_q_net = DDQN(self.state_dim, self.act_dim, dqn_cfg.hidden_size, is_cnn=False).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=getattr(dqn_cfg, "lr", 3e-4), eps=1.5e-4)

        # RL hyperparams
        self.gamma = dqn_cfg.gamma
        self.batch_size = dqn_cfg.batch_size
        self.target_update_freq = dqn_cfg.target_update_freq
        self.learning_starts = dqn_cfg.learning_starts

        # replay buffer
        self.replay_buffer = ReplayBuffer(
            dqn_cfg.replay_buffer_size,
            env_dict={
                "obs":      {"shape": (self.state_dim,), "dtype": np.float32},
                "act":      {"shape": 1, "dtype": np.int16},
                "rew":      {"shape": 1, "dtype": np.float32},
                "intr":     {"shape": 1, "dtype": np.float32},
                "term":     {"shape": 1, "dtype": np.bool_},
                "timeout":  {"shape": 1, "dtype": np.bool_},
                "next_obs": {"shape": (self.state_dim,), "dtype": np.float32},
            },
        )

        # CFN net/buffer
        self.coin_flip_dim = cfn_cfg.cfn_coin_flip_dim
        self.cfn = CoinFlipNetwork(self.cfn_state_dim, self.coin_flip_dim, device=self.device).to(self.device)
        self.cfn_optimizer = optim.RMSprop(self.cfn.parameters(), lr=1e-4, momentum=0.9, eps=1e-4, weight_decay=1e-5)
        self.cfn_buffer = CFNReplayBufferWrapper(
            size=cfn_cfg.cfn_replay_buffer_size,
            obs_shape=self.cfn_state_dim,
            coin_flip_dim=self.coin_flip_dim,
            alpha=0.5,
        )
        self.lambda_bonus = cfn_cfg.cfn_intrinsic_scale
        self.cfn_batch_size = cfn_cfg.cfn_batch_size

        # ε-sched + eval ε
        self.eps_start, self.eps_end, self.eps_decay_steps = 1.0, 0.02, 400_000
        self.eval_epsilon = 0.001
        self.epsilon = self.eps_start

        # counters
        self.step_count = 0

        # visit counts
        self.visit_counts = np.zeros((H_int, W_int), dtype=np.int32)
        self.visit_counts_all = np.zeros((H_int, W_int), dtype=np.int32)
        self.visit_counts_has_key = np.zeros((2, H_int, W_int), dtype=np.int32)
        self.visit_counts_dooropen = np.zeros((2, H_int, W_int), dtype=np.int32)
        self.visit_counts_joint = np.zeros((2, 2, H_int, W_int), dtype=np.int32)

        # plot scales
        self.POS_SCALE, self.DIR_SCALE, self.EVENT_SCALE = 80.0, 20.0, 5.0


    def _door_ahead(self, base=None):
        from minigrid.core.world_object import Door
        b = self.env.unwrapped if base is None else base
        dirs = [(1,0), (0,1), (-1,0), (0,-1)]
        dx, dy = dirs[int(getattr(b, "agent_dir", 0)) % 4]
        x, y   = map(int, b.agent_pos)
        obj = b.grid.get(x+dx, y+dy)
        return obj if isinstance(obj, Door) else None

    def _door_ahead_open(self, base=None) -> int:
        d = self._door_ahead(base)
        return int(d is not None and d.is_open)

    def _key_matches_door_ahead(self, base=None) -> int:
        d = self._door_ahead(base)
        b = self.env.unwrapped if base is None else base
        c = getattr(b, "carrying", None)
        return int(d is not None and c is not None and getattr(c, "color", None) == getattr(d, "color", None))

    def _has_key(self, base=None) -> int:
        b = self.env.unwrapped if base is None else base
        return int(getattr(b, "carrying", None) is not None)


    def cfn_compact_obs(self, base=None) -> np.ndarray:
        
        b = self.env.unwrapped if base is None else base

        # Sanity: env size should match the layout we baked into cfn_state_dim
        H_in = b.grid.height - 2
        W_in = b.grid.width  - 2
        if (H_in != self._cfn_H_in) or (W_in != self._cfn_W_in):
            raise ValueError(
                f"CFN obs size mismatch: env interior ({H_in}x{W_in}) "
                f"!= initialized ({self._cfn_H_in}x{self._cfn_W_in})."
            )

        # --- position one-hot (interior indexing) ---
        ax, ay = map(int, b.agent_pos)    # 1..W-2 / 1..H-2 in env coords
        xi = ax - 1                       # 0..W_in-1
        yi = ay - 1                       # 0..H_in-1
        pos_idx = yi * self._cfn_W_in + xi

        pos_oh = np.zeros(self._cfn_pos_dim, dtype=np.float32)
        if 0 <= pos_idx < self._cfn_pos_dim:
            pos_oh[pos_idx] = 1.0

        # --- direction one-hot (0..3) ---
        d = int(getattr(b, "agent_dir", 0)) % 4
        dir_oh = np.zeros(self._cfn_dir_dim, dtype=np.float32)
        dir_oh[d] = 1.0

        # --- binary flags as one-hot(2) each ---
        hk = int(self._has_key(b))                # 0/1
        do = int(self._door_ahead_open(b))        # 0/1
        km = int(self._key_matches_door_ahead(b)) # 0/1

        def bin_one_hot(v: int) -> np.ndarray:
            out = np.zeros(self._cfn_flag_dim, dtype=np.float32)
            out[min(max(v, 0), 1)] = 1.0
            return out

        has_key_oh   = bin_one_hot(hk)
        door_open_oh = bin_one_hot(do)
        key_match_oh = bin_one_hot(km)

        # Concatenate all one-hot parts
        return np.concatenate([pos_oh, dir_oh, has_key_oh, door_open_oh, key_match_oh], axis=0)


    def process_obs(self, obs_raw, env_for_pose=None):
        img = np.asarray(obs_raw["image"], dtype=np.int16)   # (H,W,3)
        H, W = img.shape[:2]

        # per-cell one-hot (21 features/cell)
        t = img[..., 0].clip(0, self.N_TYPE-1)
        c = img[..., 1].clip(0, self.N_COLOR-1)
        s = img[..., 2].clip(0, self.N_STATE-1)
        oh_t = self._eye_type[t]      
        oh_c = self._eye_color[c]     
        oh_s = self._eye_state[s]     
        cell_feats = np.concatenate([oh_t, oh_c, oh_s], axis=-1).reshape(-1).astype(np.float32)  # (H*W*21,)

        # agent pose from the right env
        env0 = env_for_pose if env_for_pose is not None else self.env
        ax, ay = map(int, env0.unwrapped.agent_pos)  # 1..W-2 / 1..H-2
        ax_f = ax / (W-1)                            # normalize to [0,1]
        ay_f = ay / (H-1)

        d = int(getattr(env0.unwrapped, "agent_dir", 0))  # 0..3
        dir_onehot = np.zeros(4, dtype=np.float32); dir_onehot[d] = 1.0

        return np.concatenate([cell_feats, np.array([ax_f, ay_f], np.float32), dir_onehot], 0)
    

    @torch.no_grad()
    def cfn_raw_pseudocount(self, cfn_vec: np.ndarray, update_prior=False):
        obs_t = torch.from_numpy(np.asarray(cfn_vec, dtype=np.float32)).unsqueeze(0).to(self.device)
        pred = self.cfn(obs_t, update_prior_stats=update_prior)
        norm2 = float((pred ** 2).sum()); d = float(self.coin_flip_dim)
        raw = math.sqrt(norm2 / max(d, 1.0))
        pseudocount = d / (norm2 + 1e-8)
        return raw, pseudocount

    @torch.no_grad()
    def cfn_bonus_now(self) -> float:
        cfn_vec = self.cfn_compact_obs(self.env.unwrapped)
        raw, _ = self.cfn_raw_pseudocount(cfn_vec, update_prior=True)
        return float(raw)
         

    def increment_visit_counts(self):
        base = self.env.unwrapped
        x, y = map(int, base.agent_pos)
        x -= 1; y -= 1  # interior coords

        H, W = self.visit_counts.shape
        if not (0 <= x < W and 0 <= y < H):
            return

        hk = self._has_key(base)              # int 0/1
        do = self._door_ahead_open(base)      # int 0/1

        self.visit_counts[y, x] += 1
        self.visit_counts_all[y, x] += 1
        self.visit_counts_has_key[hk, y, x]  += 1
        self.visit_counts_dooropen[do, y, x] += 1
        # if you keep a joint tensor:
        # self.visit_counts_joint[hk, do, y, x] += 1

        self.visit_counts_joint[hk, do, y, x] += 1


    def act(self, obs_tensor, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            return int(self.q_net(obs_tensor).argmax().item())

    def train(self, total_timesteps, max_episode_steps):
        current_timestep = 0
        episode_return_aug, episode_step, episode_num = 0.0, 0, 0

        obs_raw, _ = self.env.reset()
        obs = self.process_obs(obs_raw)
        self.increment_visit_counts()

        while current_timestep < total_timesteps:
            # epsilon schedule
            if current_timestep < self.eps_decay_steps:
                frac = current_timestep / self.eps_decay_steps
                self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)
            else:
                self.epsilon = self.eps_end

            # act
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.act(obs_tensor, self.epsilon)

            # step
            next_obs_raw, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = self.process_obs(next_obs_raw)
            self.increment_visit_counts()
            done = bool(terminated or truncated)

            if reward > 0:
                wandb.log({"ext_rew": reward}, step=current_timestep)

            bonus_raw = self.cfn_bonus_now()

            # if (self.step_count + 123) % 997 == 0:
            #     cfn_vec = self.cfn_compact_obs(self.env.unwrapped)
            #     raw_b, pseudo = self.cfn_raw_pseudocount(cfn_vec, update_prior=False)
            #     x, y = self.env.unwrapped.agent_pos
            wandb.log({
                f"cfn/bonus_raw": float(bonus_raw),
                # f"cfn/pseudocount)": float(pseudo),
            }, step=current_timestep)

            # store transition (uint8 obs)
            self.replay_buffer.add(
                obs=obs.astype(np.float32),
                act=int(action),
                rew=float(reward),
                intr=float(bonus_raw),
                term=bool(terminated),
                timeout=bool(truncated),
                next_obs=next_obs.astype(np.float32),
            )

            # CFN buffer
            # After you step the env and before learning:
            cfn_vec = self.cfn_compact_obs(self.env.unwrapped)
            coin_flip = get_coin_flips(self.coin_flip_dim)
            self.cfn_buffer.add(
                obs=cfn_vec.astype(np.float32),
                coin_flip=coin_flip.detach().cpu().numpy().astype(np.float32, copy=False),
                priority=1.0
            )


            # learn
            if self.replay_buffer.get_stored_size() >= max(self.batch_size, self.learning_starts):
                 self.update_q(self.replay_buffer.sample(self.batch_size), current_timestep)

            if self.cfn_buffer.get_stored_size() >= self.cfn_batch_size:
                obs_bc, coin_bc, idx = self.cfn_buffer.sample_with_indices(self.cfn_batch_size)
                self.update_cfn(obs_bc, coin_bc, current_timestep)
                self.cfn_buffer.update_priorities(idx, obs_bc, self.cfn, self.coin_flip_dim)

            # target update
            if current_timestep % self.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            # bookkeeping
            obs = next_obs
            episode_return_aug += float(reward + self.lambda_bonus * bonus_raw)
            episode_step += 1
            current_timestep += 1
            self.step_count += 1

            # episode end
            if done or episode_step >= max_episode_steps:
                wandb.log({
                    "charts/episodic_return": episode_return_aug,
                    "charts/episodic_length": episode_step,
                    "charts/episode_num": episode_num
                }, step=current_timestep)
                log.info(f"Episode {episode_num} | Steps: {episode_step} | AugReturn: {episode_return_aug:.2f} | ε: {self.epsilon:.3f} | T: {current_timestep}")
                obs_raw, _ = self.env.reset()
                obs = self.process_obs(obs_raw)
                self.increment_visit_counts()
                episode_return_aug, episode_step, episode_num = 0.0, 0, episode_num + 1

            # periodic eval/plots (once each!)
            if current_timestep % 10_000 == 0 and current_timestep > 0:
                evaluate_dqn(self, self.eval_env, current_timestep, save_dir="checkpoints_cfn", epsilon_eval=0.0)
            if current_timestep % 5000 == 0 or current_timestep == 10:
                if current_timestep % 5000 == 0:
                    plot_cfn_intrinsic_three_panels_agg(
                        self,
                        agg="mean",
                        step=current_timestep
                    )
                    

    def update_q(self, batch, step):
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        act = torch.from_numpy(batch["act"].squeeze(-1)).long().to(self.device)
        rew = torch.from_numpy(batch["rew"].squeeze(-1)).float().to(self.device)
        term = torch.from_numpy(batch["term"].squeeze(-1)).to(self.device).bool()
        intr = torch.from_numpy(batch["intr"].squeeze(-1)).float().to(self.device)
        timeout = torch.from_numpy(batch["timeout"].squeeze(-1)).to(self.device).bool()

        bootstrap_mask = (~term).float()
        aug_rew = rew + self.lambda_bonus * intr 

        q_vals = self.q_net(obs)
        q_val = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.q_net(next_obs).argmax(1)
            next_q_vals_target = self.target_q_net(next_obs)
            max_next_q = next_q_vals_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = aug_rew + bootstrap_mask * self.gamma * max_next_q
        loss = F.mse_loss(q_val, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        with torch.no_grad():
            gap = (q_vals.max(1, keepdim=True)[0] - q_vals).mean().item()

        wandb.log({
            "gap/full_mean_gap_cfn": gap,
        }, step=step)

        if self.step_count % 10000 == 0:
            wandb.log({
                "training/dqn_loss": float(loss.item()),
                "training/q_values_mean": float(q_val.mean().item()),
                "training/target_mean": float(target.mean().item()),
            }, step=step)
        
        


    def update_cfn(self, obs_batch, coin_flip_batch, step):
        obs_tensor  = obs_batch.detach().clone().to(self.device).float()      # shape: (B, 9)
        coin_tensor = coin_flip_batch.detach().clone().to(self.device).float()
        pred = self.cfn(obs_tensor, update_prior_stats=False)
        cfn_loss = F.mse_loss(pred, coin_tensor)
        self.cfn_optimizer.zero_grad(set_to_none=True)
        cfn_loss.backward()
        self.cfn_optimizer.step()
        if self.step_count % 10000 == 0:
            wandb.log({"training/cfn_loss": float(cfn_loss.item())}, step=step)

def plot_rnd_intrinsic_three_panels_agg(
    agent,
    door_pos,        # (x_d, y_d) in env coords (with walls)
    key_pos,         # (x_k, y_k) in env coords
    agg="max",       # "max" or "mean" over directions
    step=None
):
    """
    Single key-door setup:
      - Door color = 'blue'
      - Key color  = 'blue'
      - No scanning for objects; only uses provided positions.
      - For each config, sets door/key state, then sweeps all interior cells,
        aggregating CFN bonus across 4 directions (max/mean).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from minigrid.core.world_object import Door, Key

    base = agent.eval_env.unwrapped
    H_in, W_in = base.grid.height - 2, base.grid.width - 2

    # sanity
    if door_pos is None or key_pos is None:
        raise ValueError("Provide door_pos and key_pos explicitly for the single-blue setup.")

    BLUE = "blue"

    configs = [
        ("Closed, no key", dict(door_open=False, has_key=False)),
        ("Closed, has key", dict(door_open=False, has_key=True)),
        ("Open, has key",   dict(door_open=True,  has_key=True)),
    ]

    def _set_door(b, pos, is_open: bool):
        x, y = pos
        door = Door(BLUE)
        door.is_open = bool(is_open)
        door.is_locked = not door.is_open
        b.grid.set(x, y, door)

    def _place_key(b, pos):
        x, y = pos
        b.grid.set(x, y, Key(BLUE))

    def _clear_key(b, pos):
        x, y = pos
        if isinstance(b.grid.get(x, y), Key):
            b.grid.set(x, y, None)

    maps, vmin, vmax = [], +1e9, -1e9

    # freeze CFN stats while sweeping
    was_training = getattr(agent.cfn, "training", None)
    agent.cfn.eval()

    with torch.no_grad():
        for _, cfg in configs:
            agent.eval_env.reset()
            b = agent.eval_env.unwrapped

            # door state (always blue at door_pos)
            _set_door(b, door_pos, is_open=cfg["door_open"])

            # key/agent carrying state (always blue at key_pos)
            if cfg["has_key"]:
                b.carrying = Key(BLUE)
                _clear_key(b, key_pos)
            else:
                b.carrying = None
                _place_key(b, key_pos)

            # sweep interior cells; aggregate across 4 dirs
            M_dir = np.empty((4, H_in, W_in), dtype=np.float32)
            for d in range(4):
                for yy in range(H_in):
                    for xx in range(W_in):
                        b.agent_pos = (xx + 1, yy + 1)
                        b.agent_dir = d
                        cfn_vec = agent.cfn_compact_obs(b)
                        raw, _ = agent.cfn_raw_pseudocount(cfn_vec, update_prior=False)
                        M_dir[d, yy, xx] = float(raw)

            M = np.nanmax(M_dir, axis=0) if agg == "max" else np.nanmean(M_dir, axis=0)
            maps.append(M)
            if np.isfinite(M).any():
                vmin = min(vmin, float(np.nanmin(M)))
                vmax = max(vmax, float(np.nanmax(M)))

    # restore CFN mode
    if was_training is True:
        agent.cfn.train()
    elif was_training is False:
        agent.cfn.eval()

    # ---------- plot ----------
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
        ax.set_title(f"{title} ({agg} over dir)", fontsize=12, pad=6)
        ax.set_xticks(range(W_in)); ax.set_yticks(range(H_in))

        # optional visit-count overlay
        if isinstance(counts, np.ndarray) and counts.shape == (H_in, W_in):
            for yy in range(H_in):
                for xx in range(W_in):
                    ax.text(xx, yy, str(int(counts[yy, xx])),
                            ha="center", va="center", fontsize=8, color="white")

    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("CFN intrinsic (raw norm)", labelpad=6)

    if step is not None:
        import wandb as _wandb
        _wandb.log({f"heatmap/cfn_intrinsic_three_panels_{agg}": _wandb.Image(fig)}, step=step)

    # Optional scatter vs. 1/sqrt(count) (using the first panel)
    if isinstance(counts, np.ndarray) and counts.shape == (H_in, W_in) and len(maps) > 0:
        M0 = maps[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            X = 1.0 / np.sqrt(counts.astype(np.float32))
        mask = (counts > 0) & np.isfinite(X) & np.isfinite(M0)
        Xv = X[mask].ravel()
        Yv = M0[mask].ravel()
        if Xv.size > 600:
            idx = np.random.choice(Xv.size, size=600, replace=False)
            Xv, Yv = Xv[idx], Yv[idx]
        fig_sc, ax_sc = plt.subplots(figsize=(6, 6), dpi=150)
        ax_sc.scatter(Xv, Yv, s=18, alpha=0.85)
        ax_sc.set_xlabel("True Bonus  (1 / sqrt(count))")
        ax_sc.set_ylabel("Approx Bonus  (CFN)")
        ax_sc.set_title("True vs. Approx Bonus (panel: Closed, no key)")
        if step is not None:
            import wandb as _wandb
            _wandb.log({f"scatter/true_vs_approx": _wandb.Image(fig_sc)}, step=step)
        plt.close(fig_sc)

    plt.close(fig)


def probe_bonus_grid(self, env, step=None):
    """
    Compute CFN intrinsic bonus for every interior cell.
    Aggregates over 4 directions (mean).
    Logs a single heatmap to wandb.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from matplotlib import cm

    if env.unwrapped.agent_pos is None:
        env.reset()

    base = env.unwrapped
    H_int = self._cfn_H_in
    W_int = self._cfn_W_in

    bonus_grid = np.zeros((H_int, W_int), dtype=np.float32)

    # Save original agent state
    old_pos = tuple(map(int, base.agent_pos))
    old_dir = int(getattr(base, "agent_dir", 0))

    with torch.no_grad():
        for yi in range(H_int):
            for xi in range(W_int):
                base.agent_pos = (xi + 1, yi + 1)

                vals = []
                for d in range(4):
                    base.agent_dir = d
                    phi = self.cfn_compact_obs(base)
                    obs_t = torch.from_numpy(phi).unsqueeze(0).to(self.device)

                    pred = self.cfn(obs_t, update_prior_stats=False)
                    sq = (pred ** 2).sum(dim=1)
                    bonus = torch.sqrt(
                        torch.clamp(sq / float(self.coin_flip_dim), min=1e-12)
                    )

                    vals.append(float(bonus.item()))

                bonus_grid[yi, xi] = float(np.mean(vals))

    # Restore env state
    base.agent_pos = old_pos
    base.agent_dir = old_dir

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    im = ax.imshow(
        bonus_grid,
        origin="upper",
        interpolation="nearest",
        cmap=cm.viridis,
    )
    ax.set_title("CFN intrinsic bonus (interior)")
    ax.set_xticks(range(W_int))
    ax.set_yticks(range(H_int))
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()

    # ---- Log to wandb ----
    if step is not None:
        wandb.log({"cfn/bonus_heatmap": wandb.Image(fig)}, step=step)
    else:
        wandb.log({"cfn/bonus_heatmap": wandb.Image(fig)})

    plt.close(fig)

    return bonus_grid


def evaluate_dqn(agent, eval_env, step, save_dir="eval-flat", num_episodes=10,
                 log_to_wandb=True, fps=10, epsilon_eval=None):
    """
    Evaluate the agent for `num_episodes`.
    - epsilon_eval: if None, uses agent.eval_epsilon (paper gridworld: 0.001).
                    set to 0.0 for fully greedy.
    """
    os.makedirs(save_dir, exist_ok=True)
    returns, lengths = [], []

    if epsilon_eval is None:
        epsilon_eval = getattr(agent, "eval_epsilon", 0.0)

    # switch model to eval mode
    was_training = agent.q_net.training
    agent.q_net.eval()

    with torch.no_grad():
        for ep in range(num_episodes):
            obs_raw, _ = eval_env.reset()
            obs = agent.process_obs(obs_raw, env_for_pose=eval_env)

            done = False
            total_return = 0.0
            ep_len = 0

            frames = []
            pos_history = []
            record_gif = (ep == 0)
            if record_gif:
                frames.append(eval_env.render())

            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0) 
                action = agent.act(obs_t, epsilon=epsilon_eval)

                next_obs_raw, reward, terminated, truncated, _ = eval_env.step(action)
                done = bool(terminated or truncated)
                obs = agent.process_obs(next_obs_raw, env_for_pose=eval_env)

                base = eval_env.unwrapped
                pos = tuple(map(int, base.agent_pos))
                pos_history.append((pos, int(getattr(base, "agent_dir", 0)), action))

                total_return += float(reward)
                ep_len += 1

                if record_gif:
                    frames.append(eval_env.render())

            returns.append(total_return)
            lengths.append(ep_len)

            if record_gif:
                gif_path = os.path.join(save_dir, f"eval_step{step}_ep{ep}.gif")
                imageio.mimsave(gif_path, frames, fps=fps)
                if log_to_wandb:
                    import wandb
                    wandb.log({f"eval/gif_episode_{ep}": wandb.Video(gif_path, fps=fps, format="gif")}, step=step)

    # restore mode
    if was_training:
        agent.q_net.train()

    returns = np.array(returns, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.int32)
    np.savez(os.path.join(save_dir, f"eval_step_{step}.npz"), returns=returns, lengths=lengths)

    if log_to_wandb:
        import wandb
        wandb.log({
            "eval/mean_return": float(returns.mean()),
            "eval/std_return":  float(returns.std()),
            "eval/mean_length": float(lengths.mean()),
            "eval/epsilon": float(epsilon_eval),
        }, step=step)

    print(f"[EVAL] step {step} | avg return: {returns.mean():.3f} | avg len: {lengths.mean():.1f} | eps={epsilon_eval}")
    print("EVAL unique positions:", len(set(p for p,_,_ in pos_history)))
    print("EVAL first 10 steps (pos,dir,act):", pos_history[:10])
    print("EVAL last 5 steps:", pos_history[-5:])



def plot_cfn_intrinsic_three_panels_agg(agent, agg="mean", step=None):
    """
    Plot CFN intrinsic bonus heatmaps for three semantic configs:
      1) Door closed, no key
      2) Door closed, has key
      3) Door open, has key

    Aggregates CFN bonus over agent direction (mean or max).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from minigrid.core.world_object import Door, Key

    base = agent.eval_env.unwrapped
    H_in = base.grid.height - 2
    W_in = base.grid.width - 2

    # ---------- helpers ----------

    def find_primary_door(b):
        for y in range(1, b.grid.height - 1):
            for x in range(1, b.grid.width - 1):
                obj = b.grid.get(x, y)
                if isinstance(obj, Door):
                    return x, y, obj
        return None

    def remove_keys_of_color(b, color):
        for y in range(1, b.grid.height - 1):
            for x in range(1, b.grid.width - 1):
                obj = b.grid.get(x, y)
                if isinstance(obj, Key) and obj.color == color:
                    b.grid.set(x, y, None)

    # ---------- configs ----------

    configs = [
        ("Closed, no key", dict(door_open=False, has_key=False)),
        ("Closed, has key", dict(door_open=False, has_key=True)),
        ("Open, has key",   dict(door_open=True,  has_key=True)),
    ]

    maps = []

    # Freeze CFN running stats
    was_training = agent.cfn.training
    agent.cfn.eval()

    with torch.no_grad():
        for _, cfg in configs:
            agent.eval_env.reset()
            b = agent.eval_env.unwrapped

            pd = find_primary_door(b)
            door_pos, door_color = None, None
            if pd is not None:
                x_d, y_d, door = pd
                door_pos = (x_d, y_d)
                door_color = door.color

            # ---- key handling ----
            b.carrying = None
            if cfg["has_key"]:
                key_color = door_color if door_color is not None else "yellow"
                b.carrying = Key(key_color)
                remove_keys_of_color(b, key_color)

            # ---- door state ----
            if door_pos is not None:
                x_d, y_d = door_pos
                door = b.grid.get(x_d, y_d)
                door.is_open = cfg["door_open"]
                door.is_locked = not door.is_open
                b.grid.set(x_d, y_d, door)

            # ---- sweep grid ----
            M_dir = np.zeros((4, H_in, W_in), dtype=np.float32)

            for d in range(4):
                b.agent_dir = d
                for yy in range(H_in):
                    for xx in range(W_in):
                        b.agent_pos = (xx + 1, yy + 1)

                        phi = agent.cfn_compact_obs(b)
                        obs_t = torch.from_numpy(phi).unsqueeze(0).to(agent.device)

                        pred = agent.cfn(obs_t, update_prior_stats=False)
                        sq = (pred ** 2).sum(dim=1)
                        bonus = torch.sqrt(
                            torch.clamp(sq / float(agent.coin_flip_dim), min=1e-12)
                        )

                        M_dir[d, yy, xx] = float(bonus.item())

            M = np.max(M_dir, axis=0) if agg == "max" else np.mean(M_dir, axis=0)
            maps.append(M)

    # Restore CFN mode
    agent.cfn.train(was_training)

    # ---------- plot ----------
    fig, axs = plt.subplots(
        1, 4, figsize=(28, 9), dpi=200,
        gridspec_kw={"width_ratios": [1, 1, 1, 0.04]}
    )
    cax = axs[3]

    vmin = min(np.min(m) for m in maps)
    vmax = max(np.max(m) for m in maps)

    for ax, (title, _), M in zip(axs[:3], configs, maps):
        im = ax.imshow(
            M,
            cmap="viridis",
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(f"{title} ({agg} over dir)")
        ax.set_xticks(range(W_in))
        ax.set_yticks(range(H_in))

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("CFN intrinsic bonus (raw)")

    if step is not None:
        import wandb
        wandb.log(
            {f"heatmap/cfn_intrinsic_three_panels_{agg}": wandb.Image(fig)},
            step=step
        )

    plt.close(fig)



def main():
    wandb.init(project="dqn", name="cfn")

    total_timesteps   = 1_000_000
    max_episode_steps = 1600

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


    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": 512,
        "lr": 3e-4,
        "gamma": 0.99,
        "batch_size": 256,
        "replay_buffer_size": 500_000,
        "target_update_freq": 1000,
        "learning_starts": 10_000,
    })()

    cfn_cfg = type("CFNConfig", (), {
        "cfn_coin_flip_dim": 20,
        "cfn_lr": 1e-4,
        "cfn_replay_buffer_size": 500_000,
        "cfn_batch_size": 1024,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": float(np.exp(np.log(0.05/1.0) / 200_000)),
        "cfn_intrinsic_scale": 0.05,
    })()

    agent = DQN_CFNAgent(env=env, eval_env=eval_env, env_name="Fixed-DoorKey-v0", dqn_cfg=dqn_cfg, cfn_cfg=cfn_cfg)
    start = time.time()
    agent.train(total_timesteps=total_timesteps, max_episode_steps=max_episode_steps)
    end = time.time()
    print(f"Training finished in {(end - start) / 60:.2f} minutes.")

if __name__ == "__main__":
    main()
