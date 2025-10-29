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


    def update_q(self, batch, step):
        # Convert batch data to tensors
        obs = torch.from_numpy(batch["obs"]).float().to(self.device)                 
        next_obs = torch.from_numpy(batch["next_obs"]).float().to(self.device)       
        act = torch.from_numpy(batch["act"].squeeze(-1)).long().to(self.device)      
        ext = torch.from_numpy(batch["rew"].squeeze(-1)).float().to(self.device)    
        intr = torch.from_numpy(batch["intr"].squeeze(-1)).float().to(self.device)   
        term = torch.from_numpy(batch["term"].squeeze(-1)).float().to(self.device)   
        timeout = torch.from_numpy(batch["timeout"].squeeze(-1)).float().to(self.device)

        base_reward = self.ext_coef * ext + self.int_coef * self.lambda_bonus * intr  

        # Q-value for current state-action pairs
        q_s_online = self.q_net(obs)                          
        q_sa = q_s_online.gather(1, act.unsqueeze(1)).squeeze(1)  

        with torch.no_grad():
            # Log-policy terms for Munchausen update
            v_s = q_s_online.max(1, keepdim=True)[0]
            logsum_s = torch.logsumexp((q_s_online - v_s) / self.tau, dim=1, keepdim=True)
            log_pi_s = q_s_online - v_s - self.tau * logsum_s                        
            log_pi_sa = log_pi_s.gather(1, act.unsqueeze(1)).squeeze(1)             
            log_pi_sa = torch.clamp(log_pi_sa, min=self.lo, max=0.0)

        munchausen_reward = base_reward + self.alpha_m * log_pi_sa                  

        with torch.no_grad():
            # Next state value estimation using soft backup
            q_sp_online = self.q_net(next_obs)                                       
            v_sp_on = q_sp_online.max(1, keepdim=True)[0]
            logsum_sp = torch.logsumexp((q_sp_online - v_sp_on) / self.tau, dim=1, keepdim=True)
            log_pi_sp = q_sp_online - v_sp_on - self.tau * logsum_sp                
            pi_sp = F.softmax(q_sp_online / self.tau, dim=1)                         

            q_sp_target = self.target_q_net(next_obs)                                
            soft_backup = (pi_sp * (q_sp_target - self.tau * log_pi_sp)).sum(dim=1)  

            bootstrap_mask = 1.0 - term
            target = munchausen_reward + bootstrap_mask * self.gamma * soft_backup  

        # Compute loss and update parameters
        loss = F.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        # Diagnostics
        with torch.no_grad():
            gap = (q_s_online.max(1, keepdim=True)[0] - q_s_online).mean().item()
        wandb.log({
            "loss/td": loss.item(),
            "gap/full_mean_gap": gap
        }, step=step)


def dqn_cfn_mrl_main(cfg):
    wandb.init(project="dqn", name="cfn-mrl-fixed")
    total_timesteps = 1_000_000
    max_episode_steps = 400

    env = customised_doorkey.make_fixed_doorkey_env(
        size=10,
        key_color="red", key_pos=(1, 8),     
        door_color="red", door_pos=(5, 5),   
        goal_pos=(8, 1),
        agent_start_pos=(1, 1), agent_start_dir=0,
        wall_cells=[(7, 2), (8, 2)],
        # extra_keys=[((9, 1), "blue")],
        # extra_doors=[((12, 7), "blue", True)],
        ensure_door_in_wall=True,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )

    eval_env = customised_doorkey.make_fixed_doorkey_env(
        size=10,
        key_color="red", key_pos=(1, 8),     
        door_color="red", door_pos=(5, 5),   
        goal_pos=(8, 1),
        agent_start_pos=(1, 1), agent_start_dir=0,
        wall_cells=[(7, 2), (8, 2)],
        # extra_keys=[((9, 1), "blue")],
        # extra_doors=[((12, 7), "blue", True)],
        ensure_door_in_wall=True,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )

    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": 512,
        "lr": 3e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "replay_buffer_size": 500_000,
        "target_update_freq": 5000,
        "learning_starts": 20_000,
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
        tau=0.06,           
        alpha_m=0.3,      
        lo=-1.0,           
        ext_coef=2.0,      
        int_coef=1.0,      
        grad_clip=10.0,    
    )

    start = time.time()
    agent.train(total_timesteps=total_timesteps, max_episode_steps=max_episode_steps)
    end = time.time()
    print(f"Training finished in {(end - start) / 60:.2f} minutes.")


if __name__ == "__main__":
    dqn_cfn_mrl_main()