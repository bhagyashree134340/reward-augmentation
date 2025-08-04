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

from CFN.CFN import CoinFlipNetwork, CoinFlipNetworkCNN
from CFN.cfn_buffer import CFNReplayBufferWrapper
from CFN.priority_util import compute_intrinsic_reward, get_coin_flips
from utils.evaluate import evaluate_cfn_bonus_generalization, evaluate_dqn
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

        
        # tracking for debugging
        self.step_count = 0
        self.update_count = 0

    def process_obs(self, obs):
        if isinstance(obs, dict) and 'image' in obs:
            obs = obs['image']
        
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

        learning_starts = max(10000, self.batch_size * 4)

        while current_timestep < total_timesteps:
           
            # epsilon = max(self.cfn_cfg.epsilon_end, epsilon * self.cfn_cfg.epsilon_decay)
            
            action = self.act(obs, epsilon)

            next_obs_raw, reward, terminated, truncated, info = self.env.step(action)
            next_obs = self.process_obs(next_obs_raw)

            done = truncated or terminated

            if reward>0:
                wandb.log({
                    "ext_reward": reward
                }, step=current_timestep)

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)  

            with torch.no_grad():
                intrinsic_reward = compute_intrinsic_reward(
                    self.coin_flip_dim,
                    self.cfn.compute_squared_output_norm(obs_tensor)
                )
                
            intrinsic_reward_scaled = intrinsic_reward.item() * 10.0
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
                    "cfn/pseudocount": pseudocount_estimate.cpu().item(),
                }, step=current_timestep)

                
                log_intrinsic_reward_per_feature_from_obs(self, obs_tensor, step=current_timestep)

            self.cfn(obs_tensor, update_prior_stats=True)

            self.replay_buffer.append((obs, action, total_reward, next_obs, done))

            coin_flip = get_coin_flips(self.coin_flip_dim)
            self.cfn_buffer.add(obs=obs_tensor.detach().cpu().numpy(), coin_flip=coin_flip.detach().cpu().numpy(), priority=1.0)

            if current_timestep >= learning_starts and len(self.replay_buffer) >= self.batch_size:
                batch = random.sample(self.replay_buffer, self.batch_size)
                self.update(batch, current_timestep)
                self.update_count += 1

            if self.cfn_buffer.size >= self.cfn_cfg.cfn_batch_size:
                
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
                episode_return = 0
                episode_step = 0
                episode_num += 1
            
            if current_timestep % 10000 == 0 and current_timestep > 0:
                validate_dqn(self, current_timestep, save_dir="checkpoints_cfn")
                evaluate_dqn(self, self.env, current_timestep, save_dir="checkpoints_cfn")
        

    def update(self, batch, current_timestep):
        obs, act, rew, next_obs, done = map(np.array, zip(*batch))
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)

        act = torch.tensor(act, dtype=torch.long, device=self.device)
        rew = torch.tensor(rew, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

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


from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper

def log_intrinsic_reward_per_feature_from_obs(agent, obs_tensor, step=None):
    """
    Analyze intrinsic reward per object, color, and door state from a single observation.

    Args:
        agent: The RL agent with CFN and coin_flip_dim
        obs_tensor: torch.Tensor of shape (1, C, H, W), preprocessed observation
        raw_obs: np.ndarray of shape (H, W, 3), MiniGrid encoded obs with (object, color, state)
        step: Optional wandb step for logging

    Returns:
        Dictionary with average intrinsic reward per feature (object, color, state)
    """

    raw_obs = agent.env.unwrapped.grid.encode()

    # MiniGrid encodings
    OBJECT_TO_IDX = {
        "unseen": 0, "empty": 1, "wall": 2, "floor": 3, "door": 4,
        "key": 5, "ball": 6, "box": 7, "goal": 8, "lava": 9, "agent": 10,
    }
    COLOR_TO_IDX = {
        "red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5,
    }
    STATE_TO_IDX = {
        "open": 0, "closed": 1, "locked": 2,
    }

    # Inverse mappings for readability
    IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
    IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
    IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

    NUM_OBJECTS = len(OBJECT_TO_IDX)
    NUM_COLORS = len(COLOR_TO_IDX)
    NUM_STATES = len(STATE_TO_IDX)

    # Extract raw maps
    obj_map = raw_obs[:, :, 0]
    color_map = raw_obs[:, :, 1]
    state_map = raw_obs[:, :, 2]

    # Compute total intrinsic reward for this obs
    with torch.no_grad():
        output = agent.cfn(obs_tensor.to(agent.device), update_prior_stats=False)
        norm = output.norm(p=2, dim=1).item()
        intrinsic = agent.coin_flip_dim / (norm ** 2 + 1e-8)

    # Init accumulation buffers
    obj_intrinsic = np.zeros(NUM_OBJECTS)
    obj_counts = np.zeros(NUM_OBJECTS)
    color_intrinsic = np.zeros(NUM_COLORS)
    color_counts = np.zeros(NUM_COLORS)
    state_intrinsic = np.zeros(NUM_STATES)
    state_counts = np.zeros(NUM_STATES)

    # Accumulate per feature
    H, W = obj_map.shape
    for x in range(H):
        for y in range(W):
            obj_id = obj_map[x, y]
            color_id = color_map[x, y]
            state_id = state_map[x, y]

            if obj_id < NUM_OBJECTS:
                obj_intrinsic[obj_id] += intrinsic
                obj_counts[obj_id] += 1
            if color_id < NUM_COLORS:
                color_intrinsic[color_id] += intrinsic
                color_counts[color_id] += 1
            if state_id < NUM_STATES:
                state_intrinsic[state_id] += intrinsic
                state_counts[state_id] += 1

    # Average (avoid div by 0)
    obj_avg = np.divide(obj_intrinsic, obj_counts, out=np.zeros_like(obj_intrinsic), where=obj_counts > 0)
    color_avg = np.divide(color_intrinsic, color_counts, out=np.zeros_like(color_intrinsic), where=color_counts > 0)
    state_avg = np.divide(state_intrinsic, state_counts, out=np.zeros_like(state_intrinsic), where=state_counts > 0)

    # Convert to readable dict
    stats = {
        "object": {IDX_TO_OBJECT[i]: obj_avg[i] for i in range(NUM_OBJECTS) if obj_counts[i] > 0},
        "color": {IDX_TO_COLOR[i]: color_avg[i] for i in range(NUM_COLORS) if color_counts[i] > 0},
        "state": {IDX_TO_STATE[i]: state_avg[i] for i in range(NUM_STATES) if state_counts[i] > 0},
    }

    # Optional wandb log
    if step is not None:
        log_dict = {
            f"int_reward/{ftype}/{fname}": val
            for ftype, subdict in stats.items()
            for fname, val in subdict.items()
        }
        wandb.log(log_dict, step=step)

    return stats

def main():
    ENV_NAME = "MiniGrid-DoorKey-6x6-v0"  
    
    wandb.init(project="dqn", name="cfn-improved")  

    max_episode_steps = 250

    env = gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=max_episode_steps)
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)

    eval_env = gym.make(ENV_NAME, render_mode="rgb_array", max_episode_steps=max_episode_steps)
    eval_env = FullyObsWrapper(eval_env)
    eval_env = ImgObsWrapper(eval_env)
    

    total_timesteps = 1300_000

    
    hidden_size = 256  
    lr = 1e-5  
    gamma = 0.99
    batch_size = 128  
    replay_buffer_size = 1000_000  
    target_update_freq = 2000  

   
    cfn_coin_flip_dim = 20  
    cfn_lr = 0.0001  
    cfn_replay_buffer_size = 1000_000
    cfn_batch_size = 1024  
    learning_starts = 5000  
    epsilon_start = 1.0
    epsilon_end = 0.01  
    epsilon_decay = 0.999995  
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