import sys
import os

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

from CFN.CFN import CoinFlipNetwork, CoinFlipNetworkCNN
from CFN.cfn_buffer import CFNReplayBufferWrapper
from CFN.priority_util import compute_intrinsic_reward, get_coin_flips
from utils.evaluate import evaluate_cfn_bonus_generalization, evaluate_dqn
from utils.gif import save_rollout_gif
from utils.plots import plot_and_save_training_metrics, plot_eval_curve, cfn_early_vs_late_training_comparison
from utils.stats import EpisodeStats
from networks.ddqn import DDQN 
import logging
from cpprb import ReplayBuffer
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


class DQN_CFNAgent:
    def __init__(self, env, eval_env, env_name, dqn_cfg, cfn_cfg):
        self.env = env
        self.eval_env = eval_env
        self.env_name = env_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_shape = env.observation_space.shape  # (H, W, C)
        act_dim = env.action_space.n

        print("Original obs shape:", env.observation_space.shape)
        
        if len(self.obs_shape) == 3:
            self.obs_shape_torch = (self.obs_shape[2], self.obs_shape[0], self.obs_shape[1])  # (C, H, W)
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
                "act":      {"shape": 1, "dtype": np.int16},
                "rew":      {"shape": 1, "dtype": np.float32},
                "done":     {"shape": 1, "dtype": np.bool_},
                "next_obs": {"shape": self.obs_shape_torch, "dtype": np.uint8},
            },
        )

        self.cfn = CoinFlipNetworkCNN(
            obs_shape=self.obs_shape_torch,
            coin_dim=cfn_cfg.cfn_coin_flip_dim,
            device=self.device
        ).to(self.device)
        self.cfn_optimizer = optim.Adam(self.cfn.parameters(), lr=cfn_cfg.cfn_lr)

        self.cfn_buffer = CFNReplayBufferWrapper(
            size=cfn_cfg.cfn_replay_buffer_size,
            obs_shape=self.obs_shape_torch,
            coin_flip_dim=cfn_cfg.cfn_coin_flip_dim,
            alpha=0.5
        )

        self.cfn_cfg = cfn_cfg
        self.coin_flip_dim = cfn_cfg.cfn_coin_flip_dim
        self.use_cfn_prior = cfn_cfg.use_cfn_prior
        self.use_cfn_priority = cfn_cfg.use_cfn_priority

        
        # tracking for debugging
        self.step_count = 0
        self.update_count = 0

        h = self.env.unwrapped.grid.height
        w = self.env.unwrapped.grid.width
        self.visit_counts = np.zeros((h - 2, w - 2), dtype=np.int32)

    def process_obs(self, obs):
        if isinstance(obs, dict) and 'image' in obs:
            obs = obs['image']
        
        obs = np.array(obs, dtype=np.uint8)
        
        if len(obs.shape) == 3:
            obs = np.transpose(obs, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")
            
        return obs

    def increment_visit_counts(self):
        x, y = map(int, self.env.unwrapped.agent_pos)
        self.visit_counts[(y - 1), (x - 1)] += 1

    def act(self, obs, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return q_values.argmax().item()

    def train(self, total_timesteps, max_episode_steps):
        current_timestep = 0
        episode_return = 0
        episode_step = 0
        episode_num = 0
        obs_raw, _ = self.env.reset()
        obs = self.process_obs(obs_raw)
        self.increment_visit_counts()
        avg_rewards = []

        stats = EpisodeStats([], [], [])

        epsilon = self.cfn_cfg.epsilon_start

        learning_starts = max(5000, self.batch_size * 4)

        while current_timestep < total_timesteps:
           
            # epsilon = max(self.cfn_cfg.epsilon_end, epsilon * self.cfn_cfg.epsilon_decay)
            
            action = self.act(obs, epsilon)

            next_obs_raw, reward, terminated, truncated, info = self.env.step(action)
            next_obs = self.process_obs(next_obs_raw)
            self.increment_visit_counts()

            done = truncated or terminated

            if reward>0:
                wandb.log({
                    "ext_reward": reward
                }, step=current_timestep)

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            self.cfn(obs_tensor, update_prior_stats=True)

            with torch.no_grad():
                intrinsic_reward = compute_intrinsic_reward(
                    self.coin_flip_dim,
                    self.cfn.compute_squared_output_norm(obs_tensor)
                )

            intrinsic_reward_scaled = intrinsic_reward.item()

            agent_x, agent_y = map(int, self.env.unwrapped.agent_pos)
            wandb.log(
                {f"cfn/int_reward({agent_y - 1},{agent_x - 1})": intrinsic_reward_scaled},
                step=current_timestep
            )
            
            total_reward = reward + intrinsic_reward_scaled
            # avg_rewards.append(total_reward)

            if current_timestep % 1000 == 0:
                with torch.no_grad():
                    combined_out = self.cfn(obs_tensor, update_prior_stats=False)
                    output_norm = combined_out.norm(p=2, dim=1)
                    pseudocount_estimate = self.cfn.coin_flip_dim / (output_norm ** 2 + 1e-8)
                
                wandb.log({
                    "rewards/ext_reward": reward,
                    "rewards/int_reward": intrinsic_reward_scaled,
                    "rewards/total_reward": total_reward,
                    "cfn/output_norm": output_norm.cpu().item(),
                    f"cfn/pseudocount({agent_y - 1},{agent_x - 1})": pseudocount_estimate.cpu().item(),
                }, step=current_timestep)

                
                # log_intrinsic_reward_per_feature_from_obs(self, obs_tensor, step=current_timestep)


            # self.replay_buffer.append((obs, action, total_reward, next_obs, done))
            self.replay_buffer.add(
                obs=obs,
                act=action,
                rew=total_reward,
                done=done,
                next_obs=next_obs,
            )

            coin_flip = get_coin_flips(self.coin_flip_dim)
            coin_flip_np = coin_flip.detach().cpu().numpy().astype(np.float32, copy=False)
            self.cfn_buffer.add(obs=obs, coin_flip=coin_flip_np, priority=1.0)

            if current_timestep >= learning_starts and self.replay_buffer.get_stored_size() >= self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                self.update(batch, current_timestep)
                self.update_count += 1

            if self.cfn_buffer.get_stored_size() >= self.cfn_cfg.cfn_batch_size:
                
                obs_batch_bc, coin_flip_batch_bc, indices = self.cfn_buffer.sample_with_indices(self.cfn_cfg.cfn_batch_size)
                self.update_cfn(obs_batch_bc, coin_flip_batch_bc, current_timestep)

                if self.use_cfn_priority:
                    self.cfn_buffer.update_priorities(indices, obs_batch_bc, self.cfn, self.coin_flip_dim)

            obs = next_obs
            episode_return += total_reward
            episode_step += 1
            current_timestep += 1
            self.step_count += 1


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
                    f"Return: {episode_return:.2f} | Epsilon: {epsilon:.3f} | "
                    f"Total Timesteps: {current_timestep}"
                )

                if epsilon > self.cfn_cfg.epsilon_end:
                    epsilon *= self.cfn_cfg.epsilon_decay

                obs_raw, _ = self.env.reset()
                obs = self.process_obs(obs_raw)
                self.increment_visit_counts()
                episode_return = 0
                episode_step = 0
                episode_num += 1
            
            if current_timestep % 10000 == 0 and current_timestep > 0:
                
                validate_dqn(self, current_timestep, save_dir="checkpoints_cfn")
                evaluate_dqn(self, self.eval_env, current_timestep, save_dir="checkpoints_cfn")
                self.log_doorkey_true_vs_pseudo_counts(self.visit_counts, step=current_timestep)
        

    def update(self, batch, current_timestep):
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device) / 255.0
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device) / 255.0
        act = torch.from_numpy(batch["act"].squeeze(-1)).long().to(self.device)
        rew = torch.from_numpy(batch["rew"].squeeze(-1)).float().to(self.device)
        done = torch.from_numpy(batch["done"].squeeze(-1)).float().to(self.device)

        q_vals = self.q_net(obs)
        q_val = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_vals_main = self.q_net(next_obs)
            next_actions = next_q_vals_main.argmax(1)
            
            next_q_vals_target = self.target_q_net(next_obs)
            max_next_q_vals = next_q_vals_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target = rew + (1 - done) * self.gamma * max_next_q_vals

        loss = F.mse_loss(q_val, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        self.optimizer.step()
        
        if self.update_count % 1000 == 0:
            wandb.log({
                "training/dqn_loss": loss.item(),
                "training/q_values_mean": q_val.mean().item(),
                "training/target_mean": target.mean().item(),
            }, step=current_timestep)

    def update_cfn(self, obs_batch, coin_flip_batch, current_timestep):
        if isinstance(obs_batch, torch.Tensor):
            obs_tensor = obs_batch.to(self.device)
        else:
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
            
        if isinstance(coin_flip_batch, torch.Tensor):
            coin_tensor = coin_flip_batch.to(self.device)
        else:
            coin_tensor = torch.tensor(coin_flip_batch, dtype=torch.float32, device=self.device)

        pred = self.cfn(obs_tensor)
        cfn_loss = F.mse_loss(pred, coin_tensor)

        self.cfn_optimizer.zero_grad()
        cfn_loss.backward()
        self.cfn_optimizer.step()
        
        if self.update_count % 1000 == 0:
            wandb.log({
                "training/cfn_loss": cfn_loss.item(),
            }, step=current_timestep)


    def log_doorkey_true_vs_pseudo_counts(agent, visit_counts, step, save_path=None):
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        import wandb

        base_env = agent.eval_env.unwrapped

        h, w = visit_counts.shape
        true_bonus = 1.0 / np.sqrt(visit_counts + 1e-8)

        approx_bonus = np.zeros_like(true_bonus)
        mask = visit_counts == 0  # Mask for unvisited cells

        for y in range(h):
            for x in range(w):
                # if mask[y, x]:
                #     approx_bonus[y, x] = np.nan
                #     continue

                base_env.reset()
                base_env.agent_pos = (x + 1, y + 1)  
                base_env.agent_dir = np.random.randint(0, 4)  
                base_env.step(base_env.actions.toggle)  

                obs_raw = base_env.render()  
                obs = agent.process_obs(obs_raw)
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)

                if obs_tensor.max() > 1.0:
                    obs_tensor = obs_tensor / 255.0

                with torch.no_grad():
                    norm2 = agent.cfn.compute_squared_output_norm(obs_tensor).item()
                    approx_bonus[y, x] = np.sqrt(norm2 / agent.coin_flip_dim)

        true_bonus_masked = np.ma.array(true_bonus, mask=mask)
        approx_bonus_masked = np.ma.array(approx_bonus, mask=mask)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs = axs.flatten()

        # Plot True Bonus + annotate with visit counts
        im1 = axs[0].imshow(true_bonus_masked, cmap="Greens")
        axs[0].set_title("True Bonus (1/sqrt(visit_counts))")
        for i in range(h):
            for j in range(w):
                if not mask[i, j]:
                    count = visit_counts[i, j]
                    axs[0].text(j, i, f"{int(count)}", ha='center', va='center',
                                color='white' if true_bonus[i, j] < true_bonus.max() / 2 else 'black')
        plt.colorbar(im1, ax=axs[0])


        clipped_intrinsic_bonus = approx_bonus.copy()
        vmin_clip = np.nanpercentile(clipped_intrinsic_bonus, 1)
        vmax_clip = np.nanpercentile(clipped_intrinsic_bonus, 99)

        clipped_intrinsic_bonus = np.clip(clipped_intrinsic_bonus, vmin_clip, vmax_clip)
        rnd_vmin, rnd_vmax = vmin_clip, vmax_clip

        im2 = axs[1].imshow(clipped_intrinsic_bonus, cmap="viridis", vmin=rnd_vmin, vmax=rnd_vmax)
        axs[1].set_title("Intrinsic Bonus (CFN)")
        plt.colorbar(im2, ax=axs[1])

        for ax in axs:
            ax.set_xticks(range(w))
            ax.set_yticks(range(h))
            ax.set_xticklabels(range(w))
            ax.set_yticklabels(range(h))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        wandb.log({f"cfn/true_vs_intrinsic_bonus_heatmap": wandb.Image(fig)}, step=step)
        plt.close()

def main():
    # ENV_NAME = "Fixed-DoorKey-6x6-v0"  
    
    wandb.init(project="dqn", name="cfn-improved")  

    max_episode_steps = 250

    ENV_NAME = "Fixed-DoorKey-v0"

    env = gym.make(
        "Fixed-DoorKey-v0",
        size=10,
        key_pos=(1, 8),           # bottom-left room
        door_pos=(5, 5),          # middle vertical wall
        goal_pos=(8, 5),          # right room
        agent_start_pos=(1, 1),   # top-left
        agent_start_dir=0,
        disable_env_checker=True,
        max_episode_steps=400,
        render_mode="rgb_array"  # Use RGB rendering for evaluation
    )
    env = customised_doorkey.NoDropWrapper(env)
    env = FullyObsWrapper(env)
    env = RGBImgObsWrapper(env, tile_size=4)  
    env = ImgObsWrapper(env)

    eval_env = gym.make(
        "Fixed-DoorKey-v0",
        size=10,
        key_pos=(1, 8),           # bottom-left room
        door_pos=(5, 5),          # middle vertical wall
        goal_pos=(8, 5),          # right room
        agent_start_pos=(1, 1),   # top-left
        agent_start_dir=0,
        disable_env_checker=True,
        max_episode_steps=400,
        render_mode="rgb_array"  # Use RGB rendering for evaluation
    )
    eval_env = customised_doorkey.NoDropWrapper(eval_env)
    eval_env = FullyObsWrapper(eval_env)
    eval_env = RGBImgObsWrapper(eval_env, tile_size=4)
    eval_env = ImgObsWrapper(eval_env)

    max_episode_steps = 400
    total_timesteps   = 1000_000

    
    hidden_size = 256  
    lr = 3e-4
    gamma = 0.99
    batch_size = 256  
    replay_buffer_size = 1000_000  
    target_update_freq = 2000  

   
    cfn_coin_flip_dim = 20  
    cfn_lr = 0.0001  
    cfn_replay_buffer_size = 1000_000
    cfn_batch_size = 1024  
    learning_starts = 5000  
    epsilon_start = 1.0
    epsilon_end = 0.01  
    epsilon_decay = 0.99995
    use_cfn_prior = True
    use_cfn_priority = True

    print(f"Training configuration:")
    print(f"- Environment: {ENV_NAME}")
    print(f"- Total timesteps: {total_timesteps}")
    print(f"- DQN lr: {lr}, batch size: {batch_size}")
    print(f"- CFN lr: {cfn_lr}, coin dim: {cfn_coin_flip_dim}")
    print(f"- Epsilon decay: {epsilon_start} -> {epsilon_end} (decay: {epsilon_decay})")

    agent = DQN_CFNAgent(
        env=env,
        eval_env=eval_env,
        env_name=ENV_NAME,  
        dqn_cfg=type("DQNConfig", (), {
            "hidden_size": hidden_size,
            "lr": lr,
            "gamma": gamma,
            "batch_size": batch_size,
            "replay_buffer_size": replay_buffer_size,
            "target_update_freq": target_update_freq
        }),
        cfn_cfg=type("CFNConfig", (), {
            "cfn_coin_flip_dim": cfn_coin_flip_dim,
            "cfn_lr": cfn_lr,
            "cfn_replay_buffer_size": cfn_replay_buffer_size,
            "cfn_batch_size": cfn_batch_size,
            "learning_starts": learning_starts,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,
            "use_cfn_prior": use_cfn_prior,
            "use_cfn_priority": use_cfn_priority
        })
    )

    start = time.time()
    agent.train(total_timesteps=total_timesteps, max_episode_steps=max_episode_steps)
    end = time.time()
    print(f"Training finished in {(end - start) / 60:.2f} minutes.")

if __name__ == "__main__":
    main()