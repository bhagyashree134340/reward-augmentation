import math
import sys
import os

from matplotlib import pyplot as plt

from agents.dqn_rnd import RunningMeanStd
from utils.heatmaps_utils import log_small_multiples_heatmaps, plot_cfn_difficulty_panels
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import wandb
import logging
from minigrid.core.world_object import Wall

from pathlib import Path
from cpprb import ReplayBuffer
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper

import customised_doorkey
from networks.ddqn import DDQN
from CFN.CFN import CoinFlipNetworkCNN
from CFN.cfn_buffer import CFNReplayBufferWrapper
from CFN.priority_util import get_coin_flips

from utils.validate import validate_dqn
from utils.evaluate import evaluate_dqn




logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ============================== agent ========================================
class DQN_CFNAgent:
    """
    cfn per lobel et al.:
      - fresh coin flips each visit, cfn trained with mse
      - intrinsic bonus b(s) = sqrt(||f(s)||^2 / d)
      - replay stores r and b(s) separately
      - td target uses r + λ * b(s)
    """
    def __init__(self, env, eval_env, env_name, dqn_cfg, cfn_cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.int_rms = RunningMeanStd(shape=())   # scalar rewards
        self.int_clip = getattr(cfn_cfg, "cfn_int_clip", None)  # optional

        if self.use_amp:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.env = env
        self.eval_env = eval_env
        self.env_name = env_name

        # obs/act spaces
        self.obs_shape_hw_c = env.observation_space.shape  # (h,w,c)
        assert len(self.obs_shape_hw_c) == 3, f"unexpected obs shape: {self.obs_shape_hw_c}"
        self.obs_shape = (self.obs_shape_hw_c[2], self.obs_shape_hw_c[0], self.obs_shape_hw_c[1])  # (c,h,w)
        self.act_dim = env.action_space.n

        print("Original obs shape:", self.obs_shape_hw_c)
        print("PyTorch obs shape:", self.obs_shape)

        # q networks
        self.q_net = DDQN(self.obs_shape, self.act_dim, dqn_cfg.hidden_size, is_cnn=True).to(self.device)
        self.target_q_net = DDQN(self.obs_shape, self.act_dim, dqn_cfg.hidden_size, is_cnn=True).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1.25e-4, eps=1.5e-4)

        self.gamma = dqn_cfg.gamma
        self.batch_size = dqn_cfg.batch_size
        self.target_update_freq = dqn_cfg.target_update_freq
        self.learning_starts = dqn_cfg.learning_starts

        # replay (store extrinsic and intrinsic separately)
        self.replay_buffer = ReplayBuffer(
            dqn_cfg.replay_buffer_size,
            env_dict={
                "obs":      {"shape": self.obs_shape, "dtype": np.uint8},
                "act":      {"shape": 1, "dtype": np.int16},
                "rew":      {"shape": 1, "dtype": np.float32},
                "intr":     {"shape": 1, "dtype": np.float32},
                "term":     {"shape": 1, "dtype": np.bool_},  
                "timeout":  {"shape": 1, "dtype": np.bool_},   
                "next_obs": {"shape": self.obs_shape, "dtype": np.uint8},
            },
        )
        

        # cfn pieces
        self.coin_flip_dim = cfn_cfg.cfn_coin_flip_dim
        self.cfn = CoinFlipNetworkCNN(
            obs_shape=self.obs_shape,
            coin_dim=self.coin_flip_dim,
            device=self.device
        ).to(self.device)
        self.cfn_optimizer = optim.RMSprop(self.cfn.parameters(), lr=1e-4, momentum=0.9, eps=1e-4)


        self.cfn_buffer = CFNReplayBufferWrapper(
            size=cfn_cfg.cfn_replay_buffer_size,
            obs_shape=self.obs_shape,
            coin_flip_dim=self.coin_flip_dim,
            alpha=0.5  # paper blend
        )

        # intrinsic scale (λ)
        self.lambda_bonus = cfn_cfg.cfn_intrinsic_scale

        # epsilon-greedy
        self.eps_start = 1.0
        self.eps_end = 0.001
        self.eps_decay_steps = 1_000
        self.eval_epsilon = 0.001

        self.epsilon = self.eps_start

        # cfn batch size
        self.cfn_batch_size = cfn_cfg.cfn_batch_size

        # counters and visit counts (for heatmap)
        self.step_count = 0
        h = self.env.unwrapped.grid.height
        w = self.env.unwrapped.grid.width
        self.visit_counts = np.zeros((h - 2, w - 2), dtype=np.int32)

    # utils
    def process_obs(self, obs):
        # makes sure obs is numpy array, and conversts its numbers to 8-bit integers [0-255]
        x = np.asarray(obs, dtype=np.uint8) # x.shape is (h,w,c) (40, 40, 3)
        return np.transpose(x, (2, 0, 1))  

    def increment_visit_counts(self):
        x, y = map(int, self.env.unwrapped.agent_pos)
        self.visit_counts[(y - 1), (x - 1)] += 1

    @torch.no_grad()
    def _bonus_from_obs_tensor(self, obs_tensor):

        pred = self.cfn(obs_tensor, update_prior_stats=True)
        bonus_raw = (pred.norm(dim=1) / math.sqrt(self.coin_flip_dim)).item()

        
        mu  = float(np.asarray(self.int_rms.mean))
        std = float(np.sqrt(np.asarray(self.int_rms.var)) + 1e-8)
        b_norm = (bonus_raw - mu) / std

        self.int_rms.update(np.array([bonus_raw], dtype=np.float32))

        return float(b_norm)


    def act(self, obs_tensor, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
            a = int(q_values.argmax().item())
        return a

    # training loop
    def train(self, total_timesteps, max_episode_steps):
        current_timestep = 0
        episode_return_aug = 0.0  # for logging only
        episode_step = 0
        episode_num = 0

        obs_raw, _ = self.env.reset()
        obs = self.process_obs(obs_raw)
        self.increment_visit_counts()

        while current_timestep < total_timesteps:
            frac = min(1.0, current_timestep / self.eps_decay_steps)
            self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0


            # act
            action = self.act(obs_tensor, self.epsilon)

            # step env
            next_obs_raw, reward, terminated, truncated, _info = self.env.step(action)
            next_obs = self.process_obs(next_obs_raw)
            self.increment_visit_counts()
            done = bool(terminated or truncated)

            if reward > 0:
                wandb.log({
                    "ext_rew": reward
                }, step=current_timestep)

            # compute intrinsic bonus on s_t
            # next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
            bonus = self._bonus_from_obs_tensor(obs_tensor)   

            


            # if self.int_clip is not None:
            #     b_norm = np.clip(b_norm, -self.int_clip, self.int_clip)


            if current_timestep % 1000 == 0:
                wandb.log({
                    # "cfn/int_reward_raw": float(bonus),
                    "cfn/int_reward_norm": self.lambda_bonus * bonus,
                    "cfn/bonus_rms/std": float(np.sqrt(self.int_rms.var)),
                }, step=current_timestep)


            # store in rl replay
            self.replay_buffer.add(
                obs=obs,
                act=action,
                rew=float(reward),
                intr=float(bonus) if current_timestep >= self.learning_starts else 0.0,
                term=bool(terminated),
                timeout=bool(truncated),
                next_obs=next_obs,
            )

            # store in cfn replay with fresh coin flips
            coin_flip = get_coin_flips(self.coin_flip_dim)  # {-1,+1}^d
            self.cfn_buffer.add(
                obs=obs,
                coin_flip=coin_flip.detach().cpu().numpy().astype(np.float32, copy=False),
                priority=1.0
            )

            # dqn update
            if self.replay_buffer.get_stored_size() >= max(self.batch_size, self.learning_starts):
                batch = self.replay_buffer.sample(self.batch_size)
                self.update_q(batch, current_timestep)


            # cfn update
            if self.cfn_buffer.get_stored_size() >= self.cfn_batch_size:
                obs_bc, coin_bc, indices = self.cfn_buffer.sample_with_indices(self.cfn_batch_size)
                self.update_cfn(obs_bc, coin_bc, current_timestep)
                self.cfn_buffer.update_priorities(indices, obs_bc, self.cfn, self.coin_flip_dim)

            # target net update
            if current_timestep % self.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.q_net.state_dict())

            # light logging
            # if current_timestep % 1000 == 0:
            #     ax, ay = map(int, self.env.unwrapped.agent_pos)

            #     with torch.no_grad():
            #         z2 = self.cfn.compute_squared_output_norm(obs_tensor).item()  
            #         pseudo = self.coin_flip_dim / max(z2, 1e-8) 
                    
            #     wandb.log({
            #         # "charts/epsilon": float(self.epsilon),
            #         f"cfn/int_reward": float(bonus),
            #         f"cfn/pseudocount": float(pseudo),
            #         # "cfn/output_norm2": float(z2),
            #     }, step=current_timestep)

            # bookkeeping
            obs = next_obs
            episode_return_aug += float(reward + self.lambda_bonus * bonus)
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

                log.info(
                    f"Episode {episode_num} | Steps: {episode_step} | "
                    f"AugReturn(r+λB): {episode_return_aug:.2f} | ε: {self.epsilon:.3f} | "
                    f"Timestep: {current_timestep}"
                )

                obs_raw, _ = self.env.reset()
                obs = self.process_obs(obs_raw)
                self.increment_visit_counts()
                episode_return_aug = 0.0
                episode_step = 0
                episode_num += 1

            # periodic eval and heatmap
            if current_timestep % 50_000 == 0 and current_timestep > 0:
                validate_dqn(self, current_timestep, save_dir="checkpoints_cfn")
                evaluate_dqn(self, self.eval_env, current_timestep, save_dir="checkpoints_cfn", epsilon_eval=0.0)
            if current_timestep % 100_000 == 0 or current_timestep == 10:
                self.log_doorkey_true_vs_pseudo_counts(self.visit_counts, step=current_timestep)
                log_small_multiples_heatmaps(self, current_timestep, grid_h=8, grid_w=8, mask_walls=True, method_name="cfn")
                plot_cfn_difficulty_panels(self, step=current_timestep, show_counts=True, add_scatter=True)


    # q update
    def update_q(self, batch, current_timestep):
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device) / 255.0
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device) / 255.0
        act = torch.from_numpy(batch["act"].squeeze(-1)).long().to(self.device)
        rew = torch.from_numpy(batch["rew"].squeeze(-1)).float().to(self.device)
        intr = torch.from_numpy(batch["intr"].squeeze(-1)).float().to(self.device)
        term = torch.from_numpy(batch["term"].squeeze(-1)).float().to(self.device)  

        # r_aug = r + λ * b(s)
        r_aug = rew + self.lambda_bonus * intr

        # in update_q
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            q_vals = self.q_net(obs)
            q_val = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_vals_main = self.q_net(next_obs)
                next_actions = next_q_vals_main.argmax(1)
                next_q_vals_target = self.target_q_net(next_obs)
                max_next_q = next_q_vals_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                target = r_aug + (1.0 - term) * self.gamma * max_next_q

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
            }, step=current_timestep)

    # cfn update
    def update_cfn(self, obs_batch, coin_flip_batch, current_timestep):
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device) / 255.0
        coin_tensor = torch.tensor(coin_flip_batch, dtype=torch.float32, device=self.device)

        # update_cfn
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            pred = self.cfn(obs_tensor, update_prior_stats=False)   
            cfn_loss = F.mse_loss(pred, coin_tensor)

        self.cfn_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(cfn_loss).backward()
        self.scaler.step(self.cfn_optimizer)
        self.scaler.update()


        if self.step_count % 10000 == 0:
            wandb.log({"training/cfn_loss": float(cfn_loss.item())}, step=current_timestep)

    

    # heatmap (true vs intrinsic)
    from minigrid.core.world_object import Wall

    def log_doorkey_true_vs_pseudo_counts(self, visit_counts, step, save_path=None):
        
        base_env = self.eval_env.unwrapped
        # make render match training tiles
        try:
            base_env.tile_size = 4  # RGBImgObsWrapper(tile_size=4)
        except Exception:
            pass

        h, w = visit_counts.shape
        true_bonus  = 1.0 / np.sqrt(visit_counts + 1e-8)
        approx_bonus = np.full_like(true_bonus, np.nan, dtype=np.float32)  

        for y in range(h):
            for x in range(w):
                cell = base_env.grid.get(x+1, y+1)

                base_env.reset()
                base_env.agent_pos = (x + 1, y + 1)
                base_env.agent_dir = np.random.randint(0, 4)

                # one no-op step to refresh visuals
                base_env.step(base_env.actions.toggle)

                obs_raw = base_env.render()
                obs = self.process_obs(obs_raw)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0

                with torch.no_grad():
                    pred = self.cfn(obs_tensor, update_prior_stats=False)
                    approx_bonus[y, x] = float(pred.norm(p=2, dim=1) / math.sqrt(self.coin_flip_dim))


        # mask only the truly unvisited *reachable* cells for the left panel
        mask_true = visit_counts == 0
        true_bonus_masked = np.ma.array(true_bonus,  mask=mask_true)

        # do not mask the right panel so you can *see* CFN values in unvisited regions
        clipped = approx_bonus.copy()
        vmin = np.nanpercentile(clipped[~np.isnan(clipped)], 1)
        vmax = np.nanpercentile(clipped[~np.isnan(clipped)], 99)
        clipped = np.clip(clipped, vmin, vmax)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        im1 = axs[0].imshow(true_bonus_masked, cmap="Greens")
        axs[0].set_title("true bonus 1/√(visits)")
        plt.colorbar(im1, ax=axs[0])

    
        for y in range(h):
            for x in range(w):
                axs[0].text(x, y, str(int(visit_counts[y, x])),
                            ha="center", va="center", fontsize=8, color="black")
            
        im2 = axs[1].imshow(clipped, cmap="viridis", vmin=vmin, vmax=vmax)
        axs[1].set_title("intrinsic bonus (cfn)")
        plt.colorbar(im2, ax=axs[1])

        for ax in axs:
            ax.set_xticks(range(w)); ax.set_yticks(range(h))
            ax.set_xticklabels(range(w)); ax.set_yticklabels(range(h))
        plt.tight_layout()
        wandb.log({"heatmap/true_vs_intrinsic_bonus_heatmap": wandb.Image(fig)}, step=step)
        plt.close()



def main():
    wandb.init(project="dqn", name="cfn-paper-faithful")

    # env (example 10x10 fixed-doorkey)
    env = gym.make(
        "Fixed-DoorKey-v0",
        size=10,
        key_pos=(1, 8),
        door_pos=(5, 5),
        goal_pos=(8, 1),
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        disable_env_checker=True,
        max_episode_steps=400,
        render_mode="rgb_array",
    )
    env = customised_doorkey.NoDropWrapper(env)
    env = customised_doorkey.PatchGridWrapper(
        env,
        wall_cells=[(6, 1), (7, 1)],  
        goal_cell=(7,0),             
    )
    env = FullyObsWrapper(env)
    env = RGBImgObsWrapper(env, tile_size=4)
    env = ImgObsWrapper(env)

    eval_env = gym.make(
        "Fixed-DoorKey-v0",
        size=10,
        key_pos=(1, 8),
        door_pos=(5, 5),
        goal_pos=(8, 1),
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        disable_env_checker=True,
        max_episode_steps=400,
        render_mode="rgb_array",
    )
    eval_env = customised_doorkey.NoDropWrapper(eval_env)
    eval_env = customised_doorkey.PatchGridWrapper(
        eval_env,
        wall_cells=[(6, 1), (7, 1)],   
        goal_cell=None,              
    )
    eval_env = FullyObsWrapper(eval_env)
    eval_env = RGBImgObsWrapper(eval_env, tile_size=4)
    eval_env = ImgObsWrapper(eval_env)

    # hyperparams
    total_timesteps   = 1_000_000
    max_episode_steps = 200

    hidden_size = 256
    lr = 3e-4
    gamma = 0.99
    batch_size = 32
    replay_buffer_size = 500_000
    target_update_freq = 500
    learning_starts = 1_000

    cfn_coin_flip_dim = 20
    cfn_lr = 1e-4
    cfn_replay_buffer_size = 400000
    cfn_batch_size = 1024
    epsilon_start = 1.0
    epsilon_end   = 0.01
    epsilon_decay = 0.99998  

    steps_to_min = 200_000
    epsilon_start = 1.0
    epsilon_end   = 0.05
    epsilon_decay = float(np.exp(np.log(epsilon_end/epsilon_start) / steps_to_min))

    # intrinsic scale (λ) per paper
    cfn_intrinsic_scale = 0.01  # per paper: {0.001, 0.003, 0.01, 0.03}

    # pack cfgs
    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": hidden_size,
        "lr": lr,
        "gamma": gamma,
        "batch_size": batch_size,
        "replay_buffer_size": replay_buffer_size,
        "target_update_freq": target_update_freq,
        "learning_starts": learning_starts,
    })()

    cfn_cfg = type("CFNConfig", (), {
        "cfn_coin_flip_dim": cfn_coin_flip_dim,
        "cfn_lr": cfn_lr,
        "cfn_replay_buffer_size": cfn_replay_buffer_size,
        "cfn_batch_size": cfn_batch_size,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay": epsilon_decay,
        "cfn_intrinsic_scale": cfn_intrinsic_scale,
    })()

    agent = DQN_CFNAgent(
        env=env,
        eval_env=eval_env,
        env_name="Fixed-DoorKey-v0",
        dqn_cfg=dqn_cfg,
        cfn_cfg=cfn_cfg
    )

    start = time.time()
    agent.train(total_timesteps=total_timesteps, max_episode_steps=max_episode_steps)
    end = time.time()
    print(f"Training finished in {(end - start) / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
