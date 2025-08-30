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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
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
        self.state_dim = int(np.prod(self.obs_shape_hw_c)) + 3  # + (x,y,dir)

        # From FullyObsWrapper:
        H, W, _ = self.env.observation_space["image"].shape

        # MiniGrid vocab sizes
        self.N_TYPE, self.N_COLOR, self.N_STATE = 11, 6, 4
        self.FEAT_PER_CELL = self.N_TYPE + self.N_COLOR + self.N_STATE  # 21

        # One-hot caches (fast vectorized lookups)
        self._eye_type  = np.eye(self.N_TYPE,  dtype=np.float32)
        self._eye_color = np.eye(self.N_COLOR, dtype=np.float32)
        self._eye_state = np.eye(self.N_STATE, dtype=np.float32)

        # Precompute agent one-hot bases for speed (optional)
        self._agent_eye = np.eye(H*W, dtype=np.uint8).reshape(H*W, H, W)

        # New state dim: cell one-hots + agent pos one-hot + dir(sin,cos)
        # New state dim: cells + agent-pos one-hot + dir one-hot(4)
        self.state_dim = H*W*self.FEAT_PER_CELL + H*W + 4


        self.q_net = DDQN(self.state_dim, self.act_dim, dqn_cfg.hidden_size, is_cnn=False).to(self.device)
        self.target_q_net = DDQN(self.state_dim, self.act_dim, dqn_cfg.hidden_size, is_cnn=False).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1.25e-4, eps=1.5e-4)

        self.gamma = dqn_cfg.gamma
        self.batch_size = dqn_cfg.batch_size
        self.target_update_freq = dqn_cfg.target_update_freq
        self.learning_starts = dqn_cfg.learning_starts
        self.learning_starts = dqn_cfg.learning_starts

        self.replay_buffer = ReplayBuffer(
            dqn_cfg.replay_buffer_size,
            env_dict={
                "obs":      {"shape": (self.state_dim,), "dtype": np.uint8},
                "act":      {"shape": 1, "dtype": np.int16},
                "rew":      {"shape": 1, "dtype": np.float32},
                "intr":     {"shape": 1, "dtype": np.float32},
                "term":     {"shape": 1, "dtype": np.bool_},
                "timeout":  {"shape": 1, "dtype": np.bool_},
                "next_obs": {"shape": (self.state_dim,), "dtype": np.uint8},
            },
        )

        self.coin_flip_dim = cfn_cfg.cfn_coin_flip_dim
        self.cfn = CoinFlipNetwork(state_dim=self.state_dim, coin_dim=self.coin_flip_dim, device=self.device).to(self.device)
        self.cfn_optimizer = optim.RMSprop(self.cfn.parameters(), lr=1e-4, momentum=0.9, eps=1e-4, weight_decay=1e-5)
        self.cfn = CoinFlipNetwork(state_dim=self.state_dim, coin_dim=self.coin_flip_dim, device=self.device).to(self.device)
        self.cfn_buffer = CFNReplayBufferWrapper(
            size=cfn_cfg.cfn_replay_buffer_size, obs_shape=self.state_dim, coin_flip_dim=self.coin_flip_dim, alpha=0.5
        )
        self.lambda_bonus = cfn_cfg.cfn_intrinsic_scale

        self.eps_start, self.eps_end, self.eps_decay_steps = 1.0, 0.1, 500_000
        self.eval_epsilon = 0.001
        self.epsilon = self.eps_start
        self.cfn_batch_size = cfn_cfg.cfn_batch_size

        self.step_count = 0
        h = self.env.unwrapped.grid.height
        w = self.env.unwrapped.grid.width
        self.visit_counts = np.zeros((h - 2, w - 2), dtype=np.int32)

    def process_obs(self, obs_raw, env_for_pose=None):
        # img: (H, W, 3)
        img = np.asarray(obs_raw["image"], dtype=np.int16)
        t = img[..., 0].clip(0, self.N_TYPE-1)
        c = img[..., 1].clip(0, self.N_COLOR-1)
        s = img[..., 2].clip(0, self.N_STATE-1)

        oh_t = self._eye_type[t]; oh_c = self._eye_color[c]; oh_s = self._eye_state[s]
        cell_feats = np.concatenate([oh_t, oh_c, oh_s], axis=-1).reshape(-1)

        # >>> use the right env for pose <<<
        env0 = env_for_pose if env_for_pose is not None else self.env
        ax, ay = map(int, env0.unwrapped.agent_pos)
        agent_oh = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        agent_oh[ay, ax] = 1
        agent_oh = agent_oh.reshape(-1)

        d = int(getattr(env0.unwrapped, "agent_dir", 0))  # 0..3
        dir_onehot = np.zeros(4, dtype=np.uint8); dir_onehot[d] = 1

        return np.concatenate(
            [cell_feats.astype(np.float32),
            agent_oh.astype(np.float32),
            dir_onehot.astype(np.float32)], 0)

    # def process_obs(self, obs_raw):
    #     if isinstance(obs_raw, dict):
    #         img = np.asarray(obs_raw["image"], dtype=np.float32)
    #     else:
    #         img = np.asarray(obs_raw, dtype=np.float32)
    #     ax, ay = map(float, self.env.unwrapped.agent_pos)
    #     d = float(getattr(self.env.unwrapped, "agent_dir", 0))
    #     return np.concatenate([img.flatten(), np.array([ax, ay, d], np.float32)], axis=0)

    @torch.no_grad()
    def cfn_raw_pseudocount(self, obs_vec, clamp_raw=True, update_prior=False):
        if isinstance(obs_vec, np.ndarray):
            obs_t = torch.from_numpy(obs_vec).float().unsqueeze(0).to(self.device)
        else:
            obs_t = obs_vec.float().unsqueeze(0).to(self.device)
        pred = self.cfn(obs_t, update_prior_stats=update_prior)
        norm2 = float((pred ** 2).sum())
        d = float(self.coin_flip_dim)
        raw = math.sqrt(norm2 / max(d, 1.0))
        # if clamp_raw:
        #     raw = min(raw, 1.0)
        pseudocount = d / (norm2 + 1e-8)
        # pseudocount = max(pseudocount, 1.0)
        return raw, pseudocount

    @torch.no_grad()
    def _bonus_from_obs_tensor(self, obs_tensor):
        if isinstance(obs_tensor, torch.Tensor):
            obs_vec = obs_tensor.squeeze(0).detach().cpu().numpy()
        else:
            obs_vec = np.asarray(obs_tensor, dtype=np.float32)

        
        raw, _ = self.cfn_raw_pseudocount(obs_vec, clamp_raw=True, update_prior=True)
        return float(raw)

    def increment_visit_counts(self):
        x, y = map(int, self.env.unwrapped.agent_pos)
        self.visit_counts[(y - 1), (x - 1)] += 1

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
            frac = min(1.0, current_timestep / self.eps_decay_steps)
            self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

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
            bonus_raw = self._bonus_from_obs_tensor(obs_tensor)

            if (self.step_count + 123) % 997 == 0:
                raw_b, pseudo = self.cfn_raw_pseudocount(obs, clamp_raw=True, update_prior=False)
                x, y = self.env.unwrapped.agent_pos
                wandb.log({
                    f"cfn/bonus_raw({x},{y})": float(raw_b),
                    f"cfn/pseudocount({x},{y})": float(pseudo),
                }, step=current_timestep)

            # store transition (uint8 obs)
            self.replay_buffer.add(
                obs=obs.astype(np.uint8),
                act=int(action),
                rew=float(reward),
                intr=float(bonus_raw),
                term=bool(terminated),
                timeout=bool(truncated),
                next_obs=next_obs.astype(np.uint8),
            )

            # CFN buffer
            coin_flip = get_coin_flips(self.coin_flip_dim)
            self.cfn_buffer.add(
                obs=obs,
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
                plot_intrinsic_three_panels(self, current_timestep)


    def update_q(self, batch, step):
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        act = torch.from_numpy(batch["act"].squeeze(-1)).long().to(self.device)
        rew = torch.from_numpy(batch["rew"].squeeze(-1)).float().to(self.device)
        term = torch.from_numpy(batch["term"].squeeze(-1)).to(self.device).bool()
        timeout = torch.from_numpy(batch["timeout"].squeeze(-1)).to(self.device).bool()

        done = (term | timeout)
        bootstrap_mask = (~done).float()

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            q_vals = self.q_net(obs)
            q_val = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = self.q_net(next_obs).argmax(1)
                next_q_vals_target = self.target_q_net(next_obs)
                max_next_q = next_q_vals_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target = rew + bootstrap_mask * self.gamma * max_next_q
            loss = F.smooth_l1_loss(q_val, target)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.step_count % 10000 == 0:
            wandb.log({
                "training/dqn_loss": float(loss.item()),
                "training/q_values_mean": float(q_val.mean().item()),
                "training/target_mean": float(target.mean().item()),
            }, step=step)


    def update_cfn(self, obs_batch, coin_flip_batch, step):
        obs_tensor = obs_batch.detach().clone().to(self.device).float()
        coin_tensor = coin_flip_batch.detach().clone().to(self.device).float()
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            pred = self.cfn(obs_tensor, update_prior_stats=False)
            cfn_loss = F.mse_loss(pred, coin_tensor)
        self.cfn_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(cfn_loss).backward()
        self.scaler.step(self.cfn_optimizer)
        self.scaler.update()
        if self.step_count % 10000 == 0:
            wandb.log({"training/cfn_loss": float(cfn_loss.item())}, step=step)

def plot_intrinsic_three_panels(agent, step, font=12):
    base = agent.eval_env.unwrapped
    H = base.grid.height - 2
    W = base.grid.width  - 2

    def set_world(door_open=None, has_key=False):
        base.reset()
        base.carrying = None
        if has_key:
            base.carrying = Key("yellow")
        if door_open is not None:
            for y in range(1, base.grid.height-1):
                for x in range(1, base.grid.width-1):
                    obj = base.grid.get(x, y)
                    if isinstance(obj, Door):
                        obj.is_open = bool(door_open); obj.is_locked = False
                        base.grid.set(x, y, obj); return

    def obs_vec_at(x, y):
        base.agent_pos = (x+1, y+1)
        base.agent_dir = 0
        img = np.transpose(base.grid.encode(), (1,0,2))
        return agent.process_obs({"image": img}, env_for_pose=base)



    def make_map(door_open, has_key):
        set_world(door_open, has_key)
        M = np.empty((H, W), dtype=np.float32)
        for yy in range(H):
            for xx in range(W):
                raw, _ = agent.cfn_raw_pseudocount(obs_vec_at(xx, yy), clamp_raw=True, update_prior=False)
                M[yy, xx] = raw
        return M

    panels = [("Closed, no key", dict(door_open=False, has_key=False)),
              ("Closed, has key", dict(door_open=False, has_key=True)),
              ("Open, has key",   dict(door_open=True,  has_key=True))]

    maps = [make_map(**cfg) for _, cfg in panels]
    vmin, vmax = 0.0, 1.0

    fig = plt.figure(figsize=(18, 6), constrained_layout=True, dpi=120)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.04])
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    txt_pe = [pe.withStroke(linewidth=2, foreground="black")]
    last_im = None
    for ax, (title, _), M in zip(axs, panels, maps):
        last_im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
        ax.set_title(title, fontsize=font+2, pad=8)
        ax.set_xticks(range(W)); ax.set_yticks(range(H))
        ax.tick_params(labelsize=font-2)
        if hasattr(agent, "visit_counts") and getattr(agent.visit_counts, "shape", None) == (H, W):
            for yy in range(H):
                for xx in range(W):
                    ax.text(xx, yy, str(int(agent.visit_counts[yy, xx])),
                            ha="center", va="center", fontsize=font-3,
                            color="white", path_effects=txt_pe)
    if last_im is not None:
        cb = fig.colorbar(last_im, cax=cax)
        cb.ax.tick_params(labelsize=font-2)
        cb.set_label("CFN raw bonus (0–1)", fontsize=font, labelpad=8)
    try:
        wandb.log({"heatmap/intrinsic_three_panels_raw": wandb.Image(fig)}, step=step)
    except Exception:
        pass
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
                obs = agent.process_obs(obs_raw, env_for_pose=eval_env)

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


def main():
    wandb.init(project="dqn", name="cfn-paper-faithful")

    env = gym.make(
        "Fixed-DoorKey-v0", size=10,
        key_pos=(1, 8), door_pos=(5, 5), goal_pos=(8, 1),
        agent_start_pos=(1, 1), agent_start_dir=0,
        disable_env_checker=True, max_episode_steps=200, render_mode="rgb_array",
    )
    env = customised_doorkey.PatchGridWrapper(env, wall_cells=[(6, 1), (7, 1)], goal_cell=(7, 0))
    env = FullyObsWrapper(env)
    env = customised_doorkey.NoDropWrapper(env)
    

    eval_env = gym.make(
        "Fixed-DoorKey-v0", size=10,
        key_pos=(1, 8), door_pos=(5, 5), goal_pos=(8, 1),
        agent_start_pos=(1, 1), agent_start_dir=0,
        disable_env_checker=True, max_episode_steps=200, render_mode="rgb_array",
    )
    eval_env = customised_doorkey.PatchGridWrapper(eval_env, wall_cells=[(6, 1), (7, 1)], goal_cell=(7, 0))
    eval_env = FullyObsWrapper(eval_env)
    eval_env = customised_doorkey.NoDropWrapper(eval_env)


    total_timesteps   = 1_000_000
    max_episode_steps = 200

    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": 256,
        "lr": 3e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "replay_buffer_size": 500_000,
        "target_update_freq": 500,
        "learning_starts": 10_000,
    })()

    cfn_cfg = type("CFNConfig", (), {
        "cfn_coin_flip_dim": 20,
        "cfn_lr": 1e-4,
        "cfn_replay_buffer_size": 400_000,
        "cfn_batch_size": 1024,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": float(np.exp(np.log(0.05/1.0) / 200_000)),
        "cfn_intrinsic_scale": 1.0,
    })()

    agent = DQN_CFNAgent(env=env, eval_env=eval_env, env_name="Fixed-DoorKey-v0", dqn_cfg=dqn_cfg, cfn_cfg=cfn_cfg)
    start = time.time()
    agent.train(total_timesteps=total_timesteps, max_episode_steps=max_episode_steps)
    end = time.time()
    print(f"Training finished in {(end - start) / 60:.2f} minutes.")

if __name__ == "__main__":
    main()

