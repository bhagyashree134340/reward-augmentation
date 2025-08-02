import sys
import os
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

from CFN.CFN import CoinFlipNetwork, CoinFlipNetworkCNN
from CFN.cfn_buffer import CFNReplayBufferWrapper
from CFN.priority_util import compute_intrinsic_reward, get_coin_flips
from utils.evaluate import evaluate_cfn_bonus_generalization
from utils.gif import save_rollout_gif
from utils.plots import plot_and_save_training_metrics, plot_eval_curve, cfn_early_vs_late_training_comparison
from utils.stats import EpisodeStats
from networks.ddqn import DDQN 
import logging

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
        self.env_name = env_name  # Store environment name for evaluation utils
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get observation shape from environment
        self.obs_shape = env.observation_space.shape  # Should be (H, W, C) from MiniGrid
        act_dim = env.action_space.n

        print("Original obs shape:", env.observation_space.shape)
        
        # Convert to (C, H, W) format for PyTorch
        if len(self.obs_shape) == 3:
            self.obs_shape_torch = (self.obs_shape[2], self.obs_shape[0], self.obs_shape[1])  # (C, H, W)
        else:
            raise ValueError(f"Unexpected observation shape: {self.obs_shape}")
            
        print("PyTorch obs shape:", self.obs_shape_torch)

        # Initialize networks with correct observation shape
        self.q_net = DDQN(self.obs_shape_torch, act_dim, dqn_cfg.hidden_size, is_cnn=True).to(self.device)
        self.target_q_net = DDQN(self.obs_shape_torch, act_dim, dqn_cfg.hidden_size, is_cnn=True).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=dqn_cfg.lr)
        self.gamma = dqn_cfg.gamma
        self.batch_size = dqn_cfg.batch_size
        self.target_update_freq = dqn_cfg.target_update_freq

        self.replay_buffer = deque(maxlen=dqn_cfg.replay_buffer_size)

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

    def process_obs(self, obs):
        """Convert observation from (H, W, C) to (C, H, W) format"""
        if isinstance(obs, dict) and 'image' in obs:
            # Handle dict observation (some MiniGrid versions return dict)
            obs = obs['image']
        
        # Ensure it's a numpy array
        obs = np.array(obs, dtype=np.uint8)
        
        # Convert from (H, W, C) to (C, H, W)
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

    def train(self, total_timesteps, max_episode_steps):
        current_timestep = 0
        episode_return = 0
        episode_step = 0
        episode_num = 0
        obs_raw, _ = self.env.reset()
        obs = self.process_obs(obs_raw)
        avg_rewards = []

        stats = EpisodeStats([], [], [])

        epsilon = self.cfn_cfg.epsilon_start

        while current_timestep < total_timesteps:
            epsilon = max(self.cfn_cfg.epsilon_end, epsilon * self.cfn_cfg.epsilon_decay)
            action = self.act(obs, epsilon)

            next_obs_raw, reward, terminated, truncated, _ = self.env.step(action)
            next_obs = self.process_obs(next_obs_raw)

            done = truncated or terminated

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  

            intrinsic_reward = compute_intrinsic_reward(
                self.coin_flip_dim,
                self.cfn.compute_squared_output_norm(obs_tensor)
            )
            total_reward = reward + intrinsic_reward.item()
            avg_rewards.append(total_reward)

            # More frequent CFN logging to debug issues
            if current_timestep % 5000 == 0 and current_timestep > 0:  # Log every 5k steps
                with torch.no_grad():
                    combined_out = self.cfn(obs_tensor, update_prior_stats=False)
                    prior_out = self.cfn.prior(obs_tensor)
                    output_norm = combined_out.norm(p=2, dim=1)
                    prior_output_norm = prior_out.norm(p=2, dim=1)
                    pseudocount_estimate = self.cfn.coin_flip_dim / (output_norm ** 2 + 1e-8)
                    
                    # Also log epsilon for debugging
                    wandb.log({
                        "cfn/pseudocounts-intr": 1 / (intrinsic_reward.item() ** 2 + 1e-8),
                        "cfn/prior_output_norm": prior_output_norm.cpu().item(),
                        "cfn/output_norm": output_norm.cpu().item(),
                        "cfn/pseudocount_estimate": pseudocount_estimate.cpu().item(),
                        "cfn/intrinsic_reward": intrinsic_reward.item(),
                        "training/epsilon": epsilon,
                        "training/external_reward": reward,
                        "training/total_reward_avg": np.mean(avg_rewards) if avg_rewards else 0,
                    }, step=current_timestep)

            self.cfn(obs_tensor, update_prior_stats=True)

            if current_timestep % 1000 == 0:
                wandb.log({
                    "rewards/ext_reward": reward,
                    "rewards/int_reward": intrinsic_reward.item(),
                    "rewards/total_reward": total_reward,
                    "rewards/averaged_total": np.mean(avg_rewards) if avg_rewards else 0,
                    "training/epsilon": epsilon,
                    "training/episode_num": episode_num,
                }, step=current_timestep)
                avg_rewards.clear()

            self.replay_buffer.append((obs, action, total_reward, next_obs, done))

            coin_flip = get_coin_flips(self.coin_flip_dim)
            self.cfn_buffer.add(obs=obs_tensor, coin_flip=coin_flip.detach().cpu().numpy(), priority=1.0)

            if current_timestep > self.cfn_cfg.learning_starts and len(self.replay_buffer) >= self.batch_size:
                batch = random.sample(self.replay_buffer, self.batch_size)
                self.update(batch)

            # Update CFN every step for better learning
            if self.cfn_buffer.size >= self.cfn_cfg.cfn_batch_size:  # Remove the % 4 condition
                obs_batch_bc, coin_flip_batch_bc, indices = self.cfn_buffer.sample_with_indices(self.cfn_cfg.cfn_batch_size)

                self.update_cfn(obs_batch_bc, coin_flip_batch_bc)

                if self.use_cfn_priority:
                    self.cfn_buffer.update_priorities(indices, obs_batch_bc, self.cfn, self.coin_flip_dim)

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
                    f"Return: {episode_return:.2f} | Total Timesteps: {current_timestep}"
                )

                obs_raw, _ = self.env.reset()
                obs = self.process_obs(obs_raw)
                episode_return = 0
                episode_step = 0
                episode_num += 1

            # Add diagnostic logging - now uses agent's env_name
            if current_timestep % 50000 == 0 and current_timestep > 0:
                from evaluation_utils import diagnose_cfn_issues, debug_training_progress
                diagnose_cfn_issues(self)
                debug_training_progress(self)

            # Add evaluation during training - more frequent - now uses agent's env_name
            if current_timestep % 25000 == 0 and current_timestep > 0:
                print(f"\n--- Evaluation at step {current_timestep} ---")
                # Quick evaluation without GIF to save time
                from evaluation_utils import evaluate_agent_performance
                eval_metrics = evaluate_agent_performance(
                    self, num_episodes=10, max_steps=max_episode_steps
                )
                # Log to wandb
                wandb.log({
                    "eval/mean_reward": eval_metrics['mean_reward'],
                    "eval/success_rate": eval_metrics['success_rate'],
                    "eval/mean_length": eval_metrics['mean_length'],
                    "eval/std_reward": eval_metrics['std_reward'],
                }, step=current_timestep)
                
                # Create a GIF if showing progress
                if eval_metrics['success_rate'] > 0:
                    print("🎉 Creating success GIF!")
                    from evaluation_utils import create_evaluation_gif
                    create_evaluation_gif(
                        self, gif_path=f"progress_step_{current_timestep}.gif", 
                        num_episodes=2, fps=3
                    )  

        # Replace the old evaluation functions with new ones
        print("\n" + "="*50)
        print("TRAINING COMPLETED - FINAL EVALUATION")
        print("="*50)
        
        from evaluation_utils import full_evaluation, quick_test
        
        # Quick test first
        quick_metrics = quick_test(agent=self)
        
        # Full evaluation if agent shows promise
        if quick_metrics['success_rate'] > 0.2:  # If >20% success rate
            print("Agent shows promise! Running full evaluation...")
            full_metrics, episode_info = full_evaluation(agent=self)
        else:
            print("Agent needs more training, but creating a demo GIF anyway...")
            from evaluation_utils import create_evaluation_gif
            create_evaluation_gif(self, gif_path="training_demo.gif", num_episodes=3)

    def update(self, batch):
        obs, act, rew, next_obs, done = map(np.array, zip(*batch))
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)

        act = torch.tensor(act, dtype=torch.long, device=self.device)
        rew = torch.tensor(rew, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        q_vals = self.q_net(obs)
        q_val = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_vals = self.target_q_net(next_obs)
            max_next_q_vals = next_q_vals.max(1)[0]
            target = rew + (1 - done) * self.gamma * max_next_q_vals

        loss = F.mse_loss(q_val, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_cfn(self, obs_batch, coin_flip_batch):
        # Fix tensor conversion warnings
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


from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

def main1():
    # CENTRALIZED ENVIRONMENT CONFIGURATION
    ENV_NAME = "MiniGrid-DoorKey-5x5-v0"  # Change this line to switch environments
    
    wandb.init(project="dqn", name="cfn")  # Re-enable wandb logging

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    eval_env = gym.make(ENV_NAME, render_mode="rgb_array")
    eval_env = FullyObsWrapper(eval_env)
    eval_env = ImgObsWrapper(eval_env)
    max_episode_steps = 300

    total_timesteps = 500_000

    hidden_size = 128
    lr = 1e-3
    gamma = 0.99
    batch_size = 64
    replay_buffer_size = 50_000
    target_update_freq = 1000  # Less frequent updates

    # Better CFN parameters for learning
    cfn_coin_flip_dim = 64  # Increased for better exploration
    cfn_lr = 1e-3  # Higher learning rate
    cfn_replay_buffer_size = 100_000
    cfn_batch_size = 256
    learning_starts = 1000  # Start learning earlier
    epsilon_start = 1.0
    epsilon_end = 0.05  # Higher final exploration
    epsilon_decay = 0.9995  # Much slower decay
    use_cfn_prior = False
    use_cfn_priority = True

    agent = DQN_CFNAgent(
        env=env,
        eval_env=eval_env,
        env_name=ENV_NAME,  # Pass environment name to agent
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
    main1()