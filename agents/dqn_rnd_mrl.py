import torch
import torch.nn.functional as F
import wandb

from agents.dqn_rnd import DQN_RNDAgent

class RND_MRL_DQN_Agent(DQN_RNDAgent):
    """
    RND + Munchausen DQN
    - Reuses RND training/normalization/replay from DQN_RNDAgent
    - Only overrides the DQN update to use Munchausen reward shaping and soft backup
    """
    def __init__(
        self,
        env,
        eval_env,
        dqn_cfg,
        rnd_cfg,
        env_name,
        tau: float = 0.06,
        alpha_m: float = 0.3,
        lo: float = -1.0,
        ext_coef: float = None,   # if None, keep rnd_cfg.extrinsic_coef
        int_coef: float = None,   # if None, keep rnd_cfg.intrinsic_coef
        grad_clip: float = 10.0,
    ):
        super().__init__(env=env, eval_env=eval_env, dqn_cfg=dqn_cfg, rnd_cfg=rnd_cfg, env_name=env_name)
        self.tau = float(tau)
        self.alpha_m = float(alpha_m)
        self.lo = float(lo)
        self.grad_clip = float(grad_clip)

        # Allow overriding coef at agent level without changing rnd_cfg
        if ext_coef is not None:
            self.extrinsic_coef = float(ext_coef)
        if int_coef is not None:
            self.intrinsic_coef = float(int_coef)

    def update_dqn(self, batch):
    
        device = self.device
        obs      = self.obs_to_float_tensor(batch["obs"])                  
        next_obs = self.obs_to_float_tensor(batch["next_obs"])             
        act      = torch.from_numpy(batch["act"].squeeze(-1)).long().to(device)         
        ext      = torch.from_numpy(batch["ext_rew"].squeeze(-1)).float().to(device)   
        intr_n   = torch.from_numpy(batch["int_rew"].squeeze(-1)).float().to(device)    
        term     = torch.from_numpy(batch["term"].squeeze(-1)).float().to(device)       
        # timeout  = torch.from_numpy(batch["timeout"].squeeze(-1)).float().to(device)  # not used in bootstrap

        base_reward = self.extrinsic_coef * ext + self.intrinsic_coef * intr_n          

        q_s_online = self.q_net(obs)                                                   
        q_sa = q_s_online.gather(1, act.unsqueeze(1)).squeeze(1)                        

        with torch.no_grad():
            v_s = q_s_online.max(1, keepdim=True)[0]                                    
            logsum_s = torch.logsumexp((q_s_online - v_s) / self.tau, dim=1, keepdim=True)  
            log_pi_s = q_s_online - v_s - self.tau * logsum_s                           
            log_pi_sa = log_pi_s.gather(1, act.unsqueeze(1)).squeeze(1)                 
            log_pi_sa = torch.clamp(log_pi_sa, min=self.lo, max=0.0)                    

        munchausen_reward = base_reward + self.alpha_m * log_pi_sa                      

        with torch.no_grad():
            q_sp_online = self.q_net(next_obs)                                         
            v_sp = q_sp_online.max(1, keepdim=True)[0]                                  
            logsum_sp = torch.logsumexp((q_sp_online - v_sp) / self.tau, dim=1, keepdim=True)  
            log_pi_sp = q_sp_online - v_sp - self.tau * logsum_sp                      
            pi_sp = F.softmax(q_sp_online / self.tau, dim=1)                            

            q_sp_tgt = self.target_q_net(next_obs)                                     
            soft_backup = (pi_sp * (q_sp_tgt - self.tau * log_pi_sp)).sum(dim=1)        

            bootstrap_mask = 1.0 - term                                                 
            target = munchausen_reward + bootstrap_mask * self.gamma * soft_backup      

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        with torch.no_grad():
            mean_gap = (q_s_online.max(1, keepdim=True)[0] - q_s_online).mean().item()
        try:
            wandb.log({"loss/td": float(loss.item()), "gap/full_mean_gap": mean_gap})
        except Exception:
            pass


def main():
    import time
    import numpy as np
    import wandb
    from agents import customised_doorkey

    wandb.init(project="dqn", name="rnd-mrl-fixed")

    ENV_NAME = "Fixed-DoorKey-v0"
    max_episode_steps = 1600
    total_timesteps = 1_000_000

    env = customised_doorkey.make_fixed_doorkey_env(
        size=16,
        key_color="blue", key_pos=(9, 1),
        door_color="blue", door_pos=(12, 7),
        goal_pos=(9, 14),
        agent_start_pos=(1, 1), agent_start_dir=0,
        wall_cells=[(9, 7), (10, 7), (11, 7), (13, 7), (14, 7)],
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
        ensure_door_in_wall=True,
        empty_cells=[(8, 5)],
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )

    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": 1024,
        "lr": 2.5e-4,
        "gamma": 0.99,
        "batch_size": 128,
        "replay_buffer_size": 500_000,
        "target_update_freq": 2000,
    })()

    rnd_cfg = type("RNDConfig", (), {
        "intrinsic_coef": 1.0,
        "extrinsic_coef": 2.0,
        "lr": 1e-4,
        "learning_starts": 10_000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        # this agent uses linear frac decay in train(); keep for completeness
        "epsilon_decay": 0.999,
        "rnd_mask_prob": 0.5,   # predictor keep ratio / mask prob
    })()

    
    agent = RND_MRL_DQN_Agent(
        env=env,
        eval_env=eval_env,
        env_name=ENV_NAME,
        dqn_cfg=dqn_cfg,
        rnd_cfg=rnd_cfg,
        tau=0.06,        # temperature
        alpha_m=0.3,     # Munchausen weight
        lo=-1.0,         # logπ clamp lower bound
        ext_coef=2.0,    # override rnd_cfg.extrinsic_coef if desired
        int_coef=1.0,    # override rnd_cfg.intrinsic_coef if desired
        grad_clip=10.0,
    )


    start = time.time()
    agent.train(total_timesteps=total_timesteps, max_episode_steps=max_episode_steps)
    end = time.time()
    print(f"Training finished in {(end - start) / 60:.2f} minutes.")
    agent.print_visit_counts()


if __name__ == "__main__":
    main()
