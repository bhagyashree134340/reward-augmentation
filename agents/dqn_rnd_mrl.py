import time
import torch
import torch.nn.functional as F
import wandb

from agents import customised_doorkey
from minigrid.wrappers import FullyObsWrapper
import gymnasium as gym

from agents.dqn_rnd import DQN_RNDAgent

class RND_MRL_DQN_Agent(DQN_RNDAgent):
    def __init__(self, env, eval_env, dqn_cfg, rnd_cfg, env_name):
        super().__init__(env=env, eval_env=eval_env, dqn_cfg=dqn_cfg, rnd_cfg=rnd_cfg, env_name=env_name)
        # Munchausen hyperparams
        self.tau = dqn_cfg.tau
        self.alpha_m = dqn_cfg.alpha_m
        self.lo = dqn_cfg.lo

    @torch.no_grad()
    def _soft_value_next(self, q_next_target: torch.Tensor) -> torch.Tensor:
        # V_soft(s') = sum_a pi(a|s') * [ Q(s',a) - tau * log pi(a|s') ]
        log_pi_next = F.log_softmax(q_next_target / self.tau, dim=1)
        pi_next = torch.exp(log_pi_next)
        v_soft = torch.sum(pi_next * (q_next_target - self.tau * log_pi_next), dim=1)
        return v_soft

    def update_dqn(self, batch):
        # --- get tensors in your vector space ---
        obs      = self.obs_to_float_tensor(batch["obs"])
        next_obs = self.obs_to_float_tensor(batch["next_obs"])
        act      = torch.from_numpy(batch["act"].squeeze(-1)).long().to(self.device)
        ext_rew  = torch.from_numpy(batch["ext_rew"].squeeze(-1)).float().to(self.device)
        int_rew  = torch.from_numpy(batch["int_rew"].squeeze(-1)).float().to(self.device)
        term     = torch.from_numpy(batch["term"].squeeze(-1)).float().to(self.device)
        timeout  = torch.from_numpy(batch["timeout"].squeeze(-1)).float().to(self.device)

        total_rew = self.extrinsic_coef * ext_rew + self.intrinsic_coef * int_rew

        # Q(s,·) and Q(s,a)
        q_s  = self.q_net(obs)                                # (B, A)
        q_sa = q_s.gather(1, act.unsqueeze(1)).squeeze(1)     # (B,)

        # Munchausen term: alpha * clip(log pi(a|s), lo, 0)
        # Add small epsilon to prevent numerical issues
        log_pi_s = F.log_softmax(q_s / self.tau, dim=1)       # (B, A)
        log_pi_a = log_pi_s.gather(1, act.unsqueeze(1)).squeeze(1)  # (B,)
        
        # Clamp with more reasonable bounds and add small epsilon for stability
        log_pi_a_clipped = torch.clamp(log_pi_a, min=self.lo, max=-1e-8)
        r_aug = total_rew + self.alpha_m * log_pi_a_clipped

        # Bootstrap on non-terminals
        bootstrap_mask = 1.0 - term

        with torch.no_grad():
            # Use online network for action selection (Double DQN style)
            q_next_online = self.q_net(next_obs)
            next_actions = q_next_online.argmax(dim=1)
            
            # Use target network for value estimation
            q_next_tgt = self.target_q_net(next_obs)          # (B, A)
            
            # For Munchausen, we can use either soft values or selected Q-values
            # Using soft values as in your original implementation
            v_soft_next = self._soft_value_next(q_next_tgt)   # (B,)
            target = r_aug + bootstrap_mask * self.gamma * v_soft_next

        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        # Log additional metrics
        with torch.no_grad():
            wandb.log({
                "loss/td": loss.item(),
                "mrl/log_pi_a_mean": log_pi_a.mean().item(),
                "mrl/log_pi_a_std": log_pi_a.std().item(),
                "mrl/r_aug_mean": r_aug.mean().item(),
                "mrl/munchausen_bonus_mean": (self.alpha_m * log_pi_a_clipped).mean().item(),
            }, commit=False)

def main():
    wandb.init(project="dqn", name="rnd-mrl-fixed")

    ENV_NAME = "Fixed-DoorKey-v0"

    max_episode_steps = 400
    total_timesteps = 1_000_000

    env = gym.make(
        "Fixed-DoorKey-v0", size=10,
        key_pos=(1, 8), door_pos=(5, 5), goal_pos=(8, 1),
        agent_start_pos=(1, 1), agent_start_dir=0,
        disable_env_checker=True, max_episode_steps=max_episode_steps, render_mode="rgb_array",
    )
    env = customised_doorkey.PatchGridWrapper(env, wall_cells=[(6, 1), (7, 1)], goal_cell=(7, 0))
    env = FullyObsWrapper(env)
    env = customised_doorkey.NoDropWrapper(env)
    

    eval_env = gym.make(
        "Fixed-DoorKey-v0", size=10,
        key_pos=(1, 8), door_pos=(5, 5), goal_pos=(8, 1),
        agent_start_pos=(1, 1), agent_start_dir=0,
        disable_env_checker=True, max_episode_steps=max_episode_steps, render_mode="rgb_array",
    )
    eval_env = customised_doorkey.PatchGridWrapper(eval_env, wall_cells=[(6, 1), (7, 1)], goal_cell=(7, 0))
    eval_env = FullyObsWrapper(eval_env)
    eval_env = customised_doorkey.NoDropWrapper(eval_env)

    # Fixed hyperparameters - closer to working RND version
    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": 512,        # Restored to original working size
        "lr": 1e-4,                # Restored to original working learning rate
        "gamma": 0.99,
        "batch_size": 128,
        "replay_buffer_size": 1_000_000,
        "target_update_freq": 2000,
        # Munchausen parameters - more moderate values
        "tau": 0.1,                # Increased from 0.03 for more exploration
        "alpha_m": 0.5,            # Reduced from 0.9 to be less aggressive
        "lo": -5.0,                # Less restrictive bound
    })

    rnd_cfg = type("RNDConfig", (), {
        "intrinsic_coef": 1.0,
        "extrinsic_coef": 2.0,
        "lr": 1e-4,
        "learning_starts": 1000,   # Back to original faster start
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.9998,
        "rnd_mask_prob": 0.25,
    })

    agent = RND_MRL_DQN_Agent(
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