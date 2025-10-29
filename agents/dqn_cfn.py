import pathlib
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
import agents.customised_doorkey as customised_doorkey

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
log = logging.getLogger(__name__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class DQN_CFNAgent:
    def __init__(self, env, eval_env, env_name, dqn_cfg, cfn_cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp   = torch.cuda.is_available()
        self.scaler_q  = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.scaler_cfn= torch.amp.GradScaler('cuda', enabled=self.use_amp)


        if self.use_amp:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.env, self.eval_env, self.env_name = env, eval_env, env_name

        obs_space = self.env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            self.obs_shape_hw_c = obs_space["image"].shape
        else:
            self.obs_shape_hw_c = obs_space.shape
        self.obs_shape = (self.obs_shape_hw_c[2], self.obs_shape_hw_c[0], self.obs_shape_hw_c[1])
        self.act_dim = self.env.action_space.n
        
        # From FullyObsWrapper:
        H, W, _ = self.env.observation_space["image"].shape

        # MiniGrid vocab sizes
        self.N_TYPE, self.N_COLOR, self.N_STATE = 11, 6, 4
        self.FEAT_PER_CELL = self.N_TYPE + self.N_COLOR + self.N_STATE  # 21

        self.cfn_state_dim = 9  

        # One-hot caches
        self._eye_type  = np.eye(self.N_TYPE,  dtype=np.float32)
        self._eye_color = np.eye(self.N_COLOR, dtype=np.float32)
        self._eye_state = np.eye(self.N_STATE, dtype=np.float32)

        self._agent_eye = np.eye(H*W, dtype=np.uint8).reshape(H*W, H, W)

       
        self.state_dim = H*W*self.FEAT_PER_CELL + 2 + 4


        self.q_net = DDQN(self.state_dim, self.act_dim, dqn_cfg.hidden_size, is_cnn=False).to(self.device)
        self.target_q_net = DDQN(self.state_dim, self.act_dim, dqn_cfg.hidden_size, is_cnn=False).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=3e-4, eps=1.5e-4)

        self.gamma = dqn_cfg.gamma
        self.batch_size = dqn_cfg.batch_size
        self.target_update_freq = dqn_cfg.target_update_freq
        self.learning_starts = dqn_cfg.learning_starts
        self.learning_starts = dqn_cfg.learning_starts

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

        self.coin_flip_dim = cfn_cfg.cfn_coin_flip_dim
        self.cfn = CoinFlipNetwork(state_dim=self.cfn_state_dim,
                           coin_dim=self.coin_flip_dim,
                           device=self.device).to(self.device)
        self.cfn_optimizer = optim.RMSprop(self.cfn.parameters(), lr=1e-4, momentum=0.9, eps=1e-4, weight_decay=1e-5)
        self.cfn_buffer = CFNReplayBufferWrapper(
            size=cfn_cfg.cfn_replay_buffer_size,
            obs_shape=self.cfn_state_dim,                
            coin_flip_dim=self.coin_flip_dim,
            alpha=0.5
        )
        self.lambda_bonus = cfn_cfg.cfn_intrinsic_scale

        self.eps_start, self.eps_end, self.eps_decay_steps = 1.0, 0.02, 200_000  
        self.eval_epsilon = 0.001
        self.epsilon = self.eps_start
        self.cfn_batch_size = cfn_cfg.cfn_batch_size

        self.step_count = 0
        h = self.env.unwrapped.grid.height
        w = self.env.unwrapped.grid.width
        H_int = self.env.unwrapped.grid.height - 2
        W_int = self.env.unwrapped.grid.width  - 2

        self.visit_counts = np.zeros((h - 2, w - 2), dtype=np.int32)
        self.visit_counts_all       = np.zeros((H_int, W_int), dtype=np.int32)
        self.visit_counts_has_key   = np.zeros((2, H_int, W_int), dtype=np.int32)   
        self.visit_counts_dooropen  = np.zeros((2, H_int, W_int), dtype=np.int32)   
        self.visit_counts_joint     = np.zeros((2, 2, H_int, W_int), dtype=np.int32)

        self.POS_SCALE = 80.0
        self.DIR_SCALE = 20.0
        self.EVENT_SCALE = 5.0

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
        H, W = b.grid.height - 2, b.grid.width - 2
        ax, ay = map(int, b.agent_pos)
        ax_f = (ax - 1) / max(W - 1, 1)
        ay_f = (ay - 1) / max(H - 1, 1)

        d = int(getattr(b, "agent_dir", 0)) % 4
        dir_onehot = np.zeros(4, dtype=np.float32); dir_onehot[d] = 1.0

        has_key_f    = float(self._has_key(b))
        door_open_f  = float(self._door_ahead_open(b))
        key_match_f  = float(self._key_matches_door_ahead(b))

        return np.concatenate([
            np.array([ax_f, ay_f], dtype=np.float32),
            dir_onehot,
            np.array([has_key_f, door_open_f, key_match_f], dtype=np.float32)
        ], axis=0)


    def process_obs(self, obs_raw, env_for_pose=None):
        img = np.asarray(obs_raw["image"], dtype=np.int16)   
        H, W = img.shape[:2]

        # per-cell one-hot (21 features/cell)
        t = img[..., 0].clip(0, self.N_TYPE-1)
        c = img[..., 1].clip(0, self.N_COLOR-1)
        s = img[..., 2].clip(0, self.N_STATE-1)
        oh_t = self._eye_type[t]      
        oh_c = self._eye_color[c]     
        oh_s = self._eye_state[s]     
        cell_feats = np.concatenate([oh_t, oh_c, oh_s], axis=-1).reshape(-1).astype(np.float32)  

        # agent pose from the right env
        env0 = env_for_pose if env_for_pose is not None else self.env
        ax, ay = map(int, env0.unwrapped.agent_pos) 
        
        H_img, W_img = img.shape[:2]
        ax_f = (ax - 1) / max((W_img - 2), 1)
        ay_f = (ay - 1) / max((H_img - 2), 1)

        d = int(getattr(env0.unwrapped, "agent_dir", 0))  
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

        hk = self._has_key(base)             
        do = self._door_ahead_open(base)      

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

            # intrinsic (even if you don't add it to target, keep logging)
            bonus_raw = self.cfn_bonus_now()

            if (self.step_count + 123) % 997 == 0:
                cfn_vec = self.cfn_compact_obs(self.env.unwrapped)
                raw_b, pseudo = self.cfn_raw_pseudocount(cfn_vec, update_prior=False)
                x, y = self.env.unwrapped.agent_pos
                wandb.log({
                    f"cfn/bonus_raw({x},{y})": float(raw_b),
                    f"cfn/pseudocount({x},{y})": float(pseudo),
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
                plot_rnd_intrinsic_three_panels_agg(self, agg="mean", step=current_timestep)


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
        obs_tensor  = obs_batch.detach().clone().to(self.device).float()      
        coin_tensor = coin_flip_batch.detach().clone().to(self.device).float()
        pred = self.cfn(obs_tensor, update_prior_stats=False)
        cfn_loss = F.mse_loss(pred, coin_tensor)
        self.cfn_optimizer.zero_grad(set_to_none=True)
        cfn_loss.backward()
        self.cfn_optimizer.step()
        if self.step_count % 10000 == 0:
            wandb.log({"training/cfn_loss": float(cfn_loss.item())}, step=step)


def plot_rnd_intrinsic_three_panels_agg(agent, agg="max", normalized=True, step=None):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from minigrid.core.world_object import Door, Key


    def _find_primary_door(b):
        H, W = b.grid.height, b.grid.width
        primary = None
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                obj = b.grid.get(x, y)
                if isinstance(obj, Door):
                    if primary is None:
                        primary = (x, y, obj)
                    if getattr(obj, "is_locked", False):
                        return (x, y, obj)
        return primary

    def _remove_keys_of_color(b, color: str):
        H, W = b.grid.height, b.grid.width
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                obj = b.grid.get(x, y)
                if isinstance(obj, Key) and getattr(obj, "color", None) == color:
                    b.grid.set(x, y, None)

    base = agent.eval_env.unwrapped
    H_in, W_in = base.grid.height - 2, base.grid.width - 2

    configs = [
        ("Closed, no key", dict(door_open=False, has_key=False)),
        ("Closed, has key", dict(door_open=False, has_key=True)),
        ("Open, has key",   dict(door_open=True,  has_key=True)),
    ]

    maps, vmin, vmax = [], +1e9, -1e9


    was_training = getattr(agent.cfn, "training", None)
    agent.cfn.eval()
    with torch.no_grad():
        for _, cfg in configs:
            agent.eval_env.reset()
            base = agent.eval_env.unwrapped

            # Door setup
            pd = _find_primary_door(base)
            pd_pos, pd_color = None, None
            if pd is not None:
                x_d, y_d, door_obj = pd
                pd_pos = (x_d, y_d)
                pd_color = getattr(door_obj, "color", None)

            # Key / carrying setup
            base.carrying = None
            if cfg["has_key"]:
                key_color = pd_color if pd_color is not None else "yellow"
                base.carrying = Key(key_color)
                _remove_keys_of_color(base, key_color)

            # Door open/closed
            if pd_pos is not None:
                x_d, y_d = pd_pos
                obj = base.grid.get(x_d, y_d)
                if isinstance(obj, Door):
                    obj.is_open = bool(cfg["door_open"])
                    obj.is_locked = not obj.is_open
                    base.grid.set(x_d, y_d, obj)

            # Sweep over all interior cells and 4 orientations; aggregate across dirs
            M_dir = np.empty((4, H_in, W_in), dtype=np.float32)
            for d in range(4):
                for yy in range(H_in):
                    for xx in range(W_in):
                        # place agent
                        base.agent_pos = (xx + 1, yy + 1)
                        base.agent_dir = d

                        # CFN compact state for this pose/world
                        cfn_vec = agent.cfn_compact_obs(base)
                        # CFN "raw" intrinsic (same as cfn_bonus_now uses internally)
                        raw, pseudo = agent.cfn_raw_pseudocount(cfn_vec, update_prior=False)
                        M_dir[d, yy, xx] = float(raw)

            M = np.nanmax(M_dir, axis=0) if agg == "max" else np.nanmean(M_dir, axis=0)
            maps.append(M)
            if np.isfinite(M).any():
                vmin = min(vmin, float(np.nanmin(M)))
                vmax = max(vmax, float(np.nanmax(M)))

    if was_training is True:
        agent.cfn.train()
    elif was_training is False:
        agent.cfn.eval()

 
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

        
        if isinstance(counts, np.ndarray) and counts.shape == (H_in, W_in):
            for yy in range(H_in):
                for xx in range(W_in):
                    ax.text(xx, yy, str(int(counts[yy, xx])),
                            ha="center", va="center", fontsize=8, color="white")

    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("CFN intrinsic (raw norm)", labelpad=6)

    
    if step is not None:
        wandb.log(
            {f"heatmap/cfn_intrinsic_three_panels_{agg}": wandb.Image(fig)},
            step=step
        )
    if isinstance(counts, np.ndarray) and counts.shape == (H_in, W_in):
        with np.errstate(divide="ignore", invalid="ignore"):
            true_inv_sqrt = 1.0 / np.sqrt(counts.astype(np.float32))

        # mask out zero-visit and non-finite
        mask = (counts > 0) & np.isfinite(true_inv_sqrt) & np.isfinite(M)
        X = true_inv_sqrt[mask].ravel()
        Y = M[mask].ravel()

        # subsample to keep the plot clean
        if X.size > 600:
            idx = np.random.choice(X.size, size=600, replace=False)
            X, Y = X[idx], Y[idx]

        # make the scatter
        fig_sc, ax_sc = plt.subplots(figsize=(6, 6), dpi=150)
        ax_sc.scatter(X, Y, s=18, alpha=0.85)
        ax_sc.set_xlabel("True Bonus  (1 / sqrt(count))")
        ax_sc.set_ylabel("Approx Bonus  (CFN )")
        ax_sc.set_title("True vs. Approx Bonus")

        if wandb is not None and step is not None:
            wandb.log({f"scatter/true_vs_approx": wandb.Image(fig_sc)}, step=step)
        plt.close(fig_sc)

    plt.close(fig)




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


def dqn_cfn_main(cfg):
    # run = wandb.init(project="dqn", name="cfn", reinit=True)

    # code_art = wandb.Artifact(f"code-{wandb.run.id}", type="code")
    # this_file = pathlib.Path(__file__).resolve()
    # code_art.add_file(str(this_file), name=this_file.name)

    total_timesteps   = cfg.agent.training.total_timesteps
    max_episode_steps = cfg.agent.training.max_episode_steps


    env_cfg = cfg.agent.env
    env = customised_doorkey.make_fixed_doorkey_env(
        size=env_cfg.size,
        key_color=env_cfg.key_color, key_pos=env_cfg.key_pos,     
        door_color=env_cfg.door_color, door_pos=env_cfg.door_pos,   
        goal_pos=env_cfg.goal_pos,
        agent_start_pos=env_cfg.agent_start_pos, agent_start_dir=env_cfg.agent_start_dir,
        wall_cells=env_cfg.wall_cells,
        # extra_keys=[((9, 1), "blue")],
        # extra_doors=[((12, 7), "blue", True)],
        ensure_door_in_wall=env_cfg.ensure_door_in_wall,
        empty_cells=env_cfg.empty_cells,
        render_mode=env_cfg.render_mode,
        max_episode_steps=max_episode_steps,
    )

    eval_env_cfg = cfg.agent.eval_env
    eval_env = customised_doorkey.make_fixed_doorkey_env(
        size=eval_env_cfg.size,
        key_color=eval_env_cfg.key_color, key_pos=eval_env_cfg.key_pos,     
        door_color=eval_env_cfg.door_color, door_pos=eval_env_cfg.door_pos,   
        goal_pos=eval_env_cfg.goal_pos,
        agent_start_pos=eval_env_cfg.agent_start_pos, agent_start_dir=eval_env_cfg.agent_start_dir,
        wall_cells=eval_env_cfg.wall_cells,
        # extra_keys=[((9, 1), "blue")],
        # extra_doors=[((12, 7), "blue", True)],
        ensure_door_in_wall=eval_env_cfg.ensure_door_in_wall,
        empty_cells=eval_env_cfg.empty_cells,
        render_mode=eval_env_cfg.render_mode,
        max_episode_steps=max_episode_steps,
    )


    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": cfg.agent.dqn.hidden_size,
        "lr": cfg.agent.dqn.lr,
        "gamma": cfg.agent.dqn.gamma,
        "batch_size": cfg.agent.dqn.batch_size,
        "replay_buffer_size": cfg.agent.dqn.replay_buffer_size,
        "target_update_freq": cfg.agent.dqn.target_update_freq,
        "learning_starts": cfg.agent.dqn.learning_starts,
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
    dqn_cfn_main()
