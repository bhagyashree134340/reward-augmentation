import time
import torch
import wandb
from agents.dqn_rnd import DQN_RNDAgent
import torch.nn.functional as F
import gymnasium as gym
import customised_doorkey
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, RGBImgObsWrapper




class RND_MRL_DQN_Agent(DQN_RNDAgent):
    def __init__(self, env, eval_env, dqn_cfg, rnd_cfg, env_name):
        super().__init__(env=env, eval_env=eval_env, dqn_cfg=dqn_cfg, rnd_cfg=rnd_cfg, env_name=env_name)
        self.tau = dqn_cfg.tau
        self.alpha_m = dqn_cfg.alpha_m
        self.lo = dqn_cfg.lo

    @torch.no_grad()
    def _soft_value_next(self, q_next_target):
        # q_next_target: (B, A) from target network
        log_pi_next = F.log_softmax(q_next_target / self.tau, dim=1)   # (B, A)
        pi_next = torch.exp(log_pi_next)
        # V_soft(s') = Σ_a π(a|s') [ Q(s',a) - τ log π(a|s') ]
        v_soft = torch.sum(pi_next * (q_next_target - self.tau * log_pi_next), dim=1)
        return v_soft

    def update_dqn(self, batch):
        # ----- load batch (same shapes you already use) -----
        obs      = torch.tensor(batch["obs"],      dtype=torch.float32, device=self.device) / 255.0
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device) / 255.0
        act      = torch.from_numpy(batch["act"].squeeze(-1)).long().to(self.device)
        ext_rew  = torch.from_numpy(batch["ext_rew"].squeeze(-1)).float().to(self.device)
        int_rew  = torch.from_numpy(batch["int_rew"].squeeze(-1)).float().to(self.device)
        done     = torch.from_numpy(batch["done"].squeeze(-1)).float().to(self.device)

        # Mix extrinsic + intrinsic exactly as before
        total_rew = self.extrinsic_coef * ext_rew + self.intrinsic_coef * int_rew

        # Q(s,·) and chosen action value
        q_vals = self.q_net(obs)                                # (B, A)
        q_sa   = q_vals.gather(1, act.unsqueeze(1)).squeeze(1)  # (B,)

        # Munchausen reward: r' = r + α_m τ clip(log π(a|s), lo, 0)
        log_pi_s  = F.log_softmax(q_vals / self.tau, dim=1)     # (B, A)
        log_pi_a  = log_pi_s.gather(1, act.unsqueeze(1)).squeeze(1)  # (B,)
        log_pi_a  = torch.clamp(log_pi_a, min=self.lo, max=0.0)
        r_mrl     = total_rew + self.alpha_m * self.tau * log_pi_a

        with torch.no_grad():
            q_next_tgt = self.target_q_net(next_obs)            # (B, A)
            v_soft_next = self._soft_value_next(q_next_tgt)     # (B,)
            target = r_mrl + (1.0 - done) * self.gamma * v_soft_next

        loss = F.mse_loss(q_sa, target)
        # loss = F.smooth_l1_loss(q_sa, target)  # optional

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # lightweight logging
        wandb.log({
            "loss/td": loss.item(),
            "mrl/log_pi_a_mean": log_pi_a.mean().item()
        }, commit=False)


def main_mrl():
    wandb.init(project="dqn", name="rnd-mrl")
    ENV_NAME = "Fixed-DoorKey-6x6-v0"  

    env = gym.make(
        "Fixed-DoorKey-v0",
        size=16,
        disable_env_checker=True,
        render_mode="rgb_array",
        key_pos=(1, 14),
        door_pos=(8, 8),
        goal_pos=(14, 14),
        agent_start_pos=(1, 1),
    )
    env = customised_doorkey.NoDropWrapper(env)
    env = FullyObsWrapper(env)
    env = RGBImgObsWrapper(env, tile_size=4)
    env = ImgObsWrapper(env)

    eval_env = gym.make(
        "Fixed-DoorKey-v0",
        size=16,
        disable_env_checker=True,
        render_mode="rgb_array",
        key_pos=(1, 14),
        door_pos=(8, 8),
        goal_pos=(14, 14),
        agent_start_pos=(1, 1),
    )
    eval_env = customised_doorkey.NoDropWrapper(eval_env)
    eval_env = FullyObsWrapper(eval_env)
    eval_env = RGBImgObsWrapper(eval_env, tile_size=4)
    eval_env = ImgObsWrapper(eval_env)
    
    max_episode_steps = 1800
    total_timesteps = 1_300_000

    # --- DQN config with MRL params ---
    dqn_cfg = type("DQNConfig", (), {
        "hidden_size": 128,
        "lr": 1e-5,
        "gamma": 0.99,
        "batch_size": 128,
        "replay_buffer_size": 1_000_000,
        "target_update_freq": 2000,
        # Munchausen parameters
        "tau": 0.03,
        "alpha_m": 0.9,
        "lo": -1.0
    })

    rnd_cfg = type("RNDConfig", (), {
        "intrinsic_coef": 1.0,
        "extrinsic_coef": 2.0,
        "lr": 1e-4,
        "learning_starts": 10000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.9998,
        "rnd_mask_prob": 0.25
    })

    # --- Instantiate Munchausen-enabled RND DQN ---
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
    main_mrl()