import time
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from minigrid.wrappers import FullyObsWrapper
import gymnasium as gym
from agents import customised_doorkey
from agents.dqn_cfn import DQN_CFNAgent


class CFN_MRL_DQNAgent(DQN_CFNAgent):
    def __init__(
        self,
        env,
        eval_env,
        env_name=None,
        dqn_cfg=None,
        cfn_cfg=None,
        tau: float = 0.1,
        alpha_m: float = 0.03,
        lo: float = -1.0,
        ext_coef: float = 1.0,
        int_coef: float = 1.0,
        grad_clip: float = 10.0,
        **kwargs,
    ):
        super().__init__(env=env, eval_env=eval_env, env_name=env_name, dqn_cfg=dqn_cfg, cfn_cfg=cfn_cfg, **kwargs)
        self.tau = float(tau)
        self.alpha_m = float(alpha_m)
        self.lo = float(lo)
        self.ext_coef = float(ext_coef)
        self.int_coef = float(int_coef)
        self.grad_clip = float(grad_clip)

    @torch.no_grad()
    def _log_softmax_tau(self, q_values: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(q_values / self.tau, dim=1)

    @torch.no_grad()
    def _soft_value_next(self, q_next_target: torch.Tensor) -> torch.Tensor:
        log_pi_next = self._log_softmax_tau(q_next_target)
        pi_next = log_pi_next.exp()
        v_soft = torch.sum(pi_next * (q_next_target - self.tau * log_pi_next), dim=1)
        return v_soft

    def update_q(self, batch, global_step: int):
        device = self.device
        obs = torch.as_tensor(batch["obs"], device=device, dtype=torch.float32)
        actions = torch.as_tensor(batch["act"], device=device, dtype=torch.long).squeeze(-1)
        rewards = torch.as_tensor(batch["rew"], device=device, dtype=torch.float32).squeeze(-1)
        next_obs = torch.as_tensor(batch["next_obs"], device=device, dtype=torch.float32)
        term = torch.as_tensor(batch.get("term", np.zeros_like(batch["rew"])), device=device).squeeze(-1).bool()
        timeout = torch.as_tensor(batch.get("timeout", np.zeros_like(batch["rew"])), device=device).squeeze(-1).bool()
        intr_raw = torch.as_tensor(batch.get("intr", np.zeros_like(batch["rew"])), device=device, dtype=torch.float32).squeeze(-1)

        # Current Q-values
        q_all = self.q_net(obs)
        q_sa = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        q_next_tgt = self.target_q_net(next_obs)
        
        # Total reward (external + intrinsic) - no normalization
        intr_term = self.lambda_bonus * intr_raw
        total_reward = self.ext_coef * rewards + self.int_coef * intr_term

        with torch.no_grad():
            # Soft value for next state
            v_soft_next = self._soft_value_next(q_next_tgt)
            
            # For Munchausen: we need current policy on CURRENT state, not next state
            # This is the key fix - Munchausen uses current state log probabilities
            log_pi_current = F.log_softmax(q_all / self.tau, dim=1)
            log_pi_a = torch.clamp(
                log_pi_current.gather(1, actions.unsqueeze(1)).squeeze(1), 
                min=self.lo, max=0.0
            )
            
            # Munchausen RL: add tau * log_pi to the reward, not the target
            munchausen_reward = total_reward + self.alpha_m * self.tau * log_pi_a
            
            # Standard target computation
            bootstrap_mask = (~term).float()  # Don't bootstrap on timeout either
            target = munchausen_reward + self.gamma * bootstrap_mask * v_soft_next

        # Loss and optimization
        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Logging
        if (global_step % 1000) == 0:
            try:
                top2 = q_all.topk(2, dim=1).values
                gap = (top2[:, 0] - top2[:, 1]).mean().item()
                wandb.log({
                    "loss/q": loss.item(),
                    "munchausen/log_pi_current_mean": log_pi_a.mean().item(),
                    "munchausen/munchausen_reward_mean": munchausen_reward.mean().item(),
                    "intrinsic/cfn_raw_mean": intr_raw.mean().item(),
                    "diagnostics/action_gap": gap,
                }, step=global_step)
            except Exception:
                pass


def main():
    wandb.init(project="dqn", name="cfn-mrl-fixed")
    total_timesteps = 1_000_000
    max_episode_steps = 400

    env = gym.make(
        "Fixed-DoorKey-v0",
        size=10,
        key_pos=(1, 8),
        door_pos=(5, 5),
        goal_pos=(8, 1),
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        disable_env_checker=True,
        max_episode_steps=max_episode_steps,
        render_mode="rgb_array",
    )
    env = customised_doorkey.PatchGridWrapper(env, wall_cells=[(6, 1), (7, 1)], goal_cell=(7, 0))
    env = FullyObsWrapper(env)
    env = customised_doorkey.NoDropWrapper(env)

    eval_env = gym.make(
        "Fixed-DoorKey-v0",
        size=10,
        key_pos=(1, 8),
        door_pos=(5, 5),
        goal_pos=(8, 1),
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        disable_env_checker=True,
        max_episode_steps=max_episode_steps,
        render_mode="rgb_array",
    )
    eval_env = customised_doorkey.PatchGridWrapper(eval_env, wall_cells=[(6, 1), (7, 1)], goal_cell=(7, 0))
    eval_env = FullyObsWrapper(eval_env)
    eval_env = customised_doorkey.NoDropWrapper(eval_env)

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
        "cfn_intrinsic_scale": 0.005,
    })()

    agent = CFN_MRL_DQNAgent(
        env=env,
        eval_env=eval_env,
        env_name="Fixed-DoorKey-v0",
        dqn_cfg=dqn_cfg,
        cfn_cfg=cfn_cfg,
        tau=0.1,           # Conservative temperature
        alpha_m=0.03,      # Conservative Munchausen coefficient  
        lo=-1.0,           
        ext_coef=1.0,      
        int_coef=0.0,      # Start without intrinsic rewards
        grad_clip=10.0,    
    )

    start = time.time()
    agent.train(total_timesteps=total_timesteps, max_episode_steps=max_episode_steps)
    end = time.time()
    print(f"Training finished in {(end - start) / 60:.2f} minutes.")


if __name__ == "__main__":
    main()