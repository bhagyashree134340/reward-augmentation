
import logging
import torch
import torch.nn.functional as F
import numpy as np
from RND.rnd import RNDModel, RewardForwardFilter
from gymnasium.wrappers.utils import RunningMeanStd
import torch.nn.functional as F
from agents.sac_agent import SACAgent
from utils.evaluate import evaluate
from utils.validate import validate

log = logging.getLogger(__name__)

class SACRNDAgent(SACAgent):
    def __init__(self, env, eval_env, rnd_cfg, **kwargs):
        super().__init__(env, eval_env, **kwargs)

        self.rnd_cfg = rnd_cfg
        self.intrinsic_coef = rnd_cfg.intrinsic_coef
        self.extrinsic_coef = rnd_cfg.extrinsic_coef

        obs_dim = env.observation_space.shape[0]
        self.rnd = RNDModel(obs_dim).to(self.device)
        self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(), lr=rnd_cfg.lr)

        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.reward_rms = RunningMeanStd()
        self.reward_filter = RewardForwardFilter(gamma=0.99)

    def train(self, total_timesteps: int, max_episode_steps: int) -> None:
        obs, _ = self.env.reset()
        current_timestep = 0
        episode_return = 0
        episode_step = 0

        while current_timestep < total_timesteps:
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                action, _ = self.actor(obs)
                action = action.cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)

            next_obs, ext_reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            self.obs_rms.update(next_obs_tensor.numpy())
            norm_obs = (next_obs_tensor - torch.from_numpy(self.obs_rms.mean).float()) / \
                       torch.sqrt(torch.from_numpy(self.obs_rms.var).float() + 1e-8)

            with torch.no_grad():
                target = self.rnd.target(norm_obs.to(self.device))
            pred = self.rnd.predictor(norm_obs.to(self.device))
            int_reward = 0.5 * ((pred - target) ** 2).sum().item()

            discounted_r = self.reward_filter.update(int_reward)
            self.reward_rms.update(np.array([discounted_r]))
            norm_int_reward = int_reward / np.sqrt(np.maximum(self.reward_rms.var, 1e-8))

            total_reward = self.extrinsic_coef * ext_reward + self.intrinsic_coef * norm_int_reward

            self.buffer.add(
            obs=obs.cpu().numpy().astype(np.float32),
            act=np.array(action, dtype=np.float32),
            ext_rew=np.array(ext_reward, dtype=np.float32),
            int_rew=np.array(norm_int_reward, dtype=np.float32),
            next_obs=np.array(next_obs, dtype=np.float32),
            done=np.array(done, dtype=np.float32)
)

            mask_prob = self.rnd_cfg.rnd_mask_prob if hasattr(self.rnd_cfg, "rnd_mask_prob") else 0.25
            mask = torch.rand_like(pred[:, 0]) < mask_prob 
            if mask.sum() > 0:
                forward_loss = F.mse_loss(pred[mask], target.detach()[mask])
                self.rnd_optimizer.zero_grad()
                forward_loss.backward()
                self.rnd_optimizer.step()

            if self.buffer.get_stored_size() >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)

                obs_batch = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
                act_batch = torch.tensor(batch["act"], dtype=torch.float32, device=self.device)
                ext_rew_batch = torch.tensor(batch["ext_rew"], dtype=torch.float32, device=self.device).squeeze(-1)
                int_rew_batch = torch.tensor(batch["int_rew"], dtype=torch.float32, device=self.device).squeeze(-1)
                next_obs_batch = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
                done_batch = torch.tensor(batch["done"], dtype=torch.float32, device=self.device).squeeze(-1)

                total_rew_batch = self.extrinsic_coef * ext_rew_batch + self.intrinsic_coef * int_rew_batch

                self.update(obs_batch, act_batch, total_rew_batch, next_obs_batch, done_batch, current_step=current_timestep)

            obs = next_obs
            current_timestep += 1
            episode_step += 1
            episode_return += ext_reward

            if done or episode_step >= max_episode_steps:
                log.info(
                    f"Episode done | Steps: {episode_step} | "
                    f"Return: {episode_return:.2f} | Total Timesteps: {current_timestep}"
                )
                
                obs, _ = self.env.reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                episode_return = 0
                episode_step = 0

            if current_timestep % 1000 == 0:
                validate(self.actor, current_timestep)

                eval_envstep, eval_mean, eval_std = evaluate(self.actor, self.eval_env, current_timestep, max_episode_steps)
                self.eval_envsteps.append(eval_envstep)
                self.eval_means.append(eval_mean)
                self.eval_stds.append(eval_std)
