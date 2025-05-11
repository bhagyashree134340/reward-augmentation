from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
import wandb
from cpprb import ReplayBuffer
from hydra.core.hydra_config import HydraConfig
from networks.network import Critic, Actor
from utils.evaluate import evaluate_and_log
from utils.plots import plot_and_save_training_metrics, plot_eval_curve
from utils.polyak import polyak_update
import logging

from utils.stats import EpisodeStats

log = logging.getLogger(__name__)


class SACAgent:
    def __init__(self,
                 env,
                 gamma=0.99,
                 lr=0.001,
                 batch_size=64,
                 tau=0.005,
                 maxlen=100_000,
                 target_entropy=-1.0
                 ):
        """
        Initialize the SAC agent.

        :param env: The environment.
        :param exploration_noise.
        :param gamma: The discount factor.
        :param lr: The learning rate.
        :param batch_size: Mini batch size.
        :param tau: Polyak update coefficient.
        :param max_size: Maximum number of transitions in the buffer.
        """

        self.eval_envsteps, self.eval_means, self.eval_stds = [], [], []
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.target_entropy = -np.prod(env.action_space.shape).item()

        # Initialize the Replay Buffer
        # self.buffer = ReplayBuffer(maxlen)
        self.buffer = ReplayBuffer(
            size=maxlen,
            env_dict={
                "obs": {"shape": env.observation_space.shape[0]},
                "act": {"shape": env.action_space.shape[0]},
                "ext_rew": {},
                "int_rew": {},
                "next_obs": {"shape": env.observation_space.shape[0]},
                "done": {}
            }
        )

        # Initialize two critic and one actor network
        self.q1 = Critic(env.observation_space.shape[0], env.action_space.shape[0])
        self.q2 = Critic(env.observation_space.shape[0], env.action_space.shape[0])
        self.actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low,
                           env.action_space.high)
        self.log_ent_coef = torch.zeros(1, requires_grad=True)

        # Initialze two target critic and one target actor networks and load the corresponding state_dicts
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.actor_target = copy.deepcopy(self.actor)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Create ADAM optimizer for the Critic and Actor networks
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.ent_coef_optimizer = optim.Adam([self.log_ent_coef], lr=lr)

    def train(self, total_timesteps: int, max_steps: int) -> None:
        """
        Train the SAC agent using a timestep-based loop.

        :param total_timesteps: Total number of environment steps to train for.
        :param max_steps: Max steps per episode (for truncation only).
        """
        current_timestep = 0
        episode_return = 0
        episode_step = 0
        episode_num = 0
        episode_lengths = []
        episode_rewards = []
        timesteps_on_ep_end = []

        obs, _ = self.env.reset()

        while current_timestep < total_timesteps:
            # self.cfn.prior.train()
            # Select action
            with torch.no_grad():
                action, _ = self.actor(torch.as_tensor(obs).float())
                action = action.cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)

            done = terminated or truncated

            wandb.log({
                "ext_reward": reward
            })

            self.buffer.add(
                obs=np.array(obs, dtype=np.float32),
                act=np.array(action, dtype=np.float32),
                ext_rew=np.array(reward, dtype=np.float32),
                int_rew=np.array(0, dtype=np.float32),
                next_obs=np.array(next_obs, dtype=np.float32),
                done=np.array(done, dtype=np.float32)
            )

            # Sample and update
            if self.buffer.get_stored_size() >= self.batch_size:
                # obs_batch, act_batch, rew_batch, next_obs_batch, tm_batch = self.buffer.sample(self.batch_size)

                sample = self.buffer.sample(self.batch_size)

                obs_batch = torch.tensor(sample["obs"], dtype=torch.float32)
                act_batch = torch.tensor(sample["act"], dtype=torch.float32)
                ext_rew_batch = torch.tensor(sample["ext_rew"], dtype=torch.float32).squeeze(-1)
                # int_rew_batch = torch.tensor(sample["int_rew"], dtype=torch.float32).squeeze(-1)
                next_obs_batch = torch.tensor(sample["next_obs"], dtype=torch.float32)
                tm_batch = torch.tensor(sample["done"], dtype=torch.float32).squeeze(-1)

                self.update(obs_batch, act_batch, ext_rew_batch, next_obs_batch, tm_batch)

            obs = next_obs
            episode_return += reward
            episode_step += 1
            current_timestep += 1

            if done or episode_step >= max_steps:
                episode_lengths.append(episode_step)
                episode_rewards.append(episode_return)
                timesteps_on_ep_end.append(current_timestep)

                wandb.log({
                    "episode return": episode_return,
                    "episode length": episode_step,
                    "episode num": episode_num
                    # "intrinsic rewards": intrinsic_reward,
                    # "reward": reward
                }, step=current_timestep)

                log.info(
                    f"Episode {episode_num} | Steps: {episode_step} | "
                    f"Return: {episode_return:.2f} | Total Timesteps: {current_timestep}"
                )

                obs, _ = self.env.reset()
                episode_return = 0
                episode_step = 0
                episode_num += 1

            # Save & evaluate periodically
            if current_timestep % 1000 == 0:
                eval_envstep, eval_mean, eval_std = evaluate_and_log(self.actor, self.env, current_timestep, max_steps)
                self.eval_envsteps.append(eval_envstep)
                self.eval_means.append(eval_mean)
                self.eval_stds.append(eval_std)

        stats = EpisodeStats(
            episode_lengths=episode_lengths,
            episode_rewards=episode_rewards,
            timesteps_on_ep_end=timesteps_on_ep_end
        )

        plot_and_save_training_metrics(
            stats,
            output_dir=Path(HydraConfig.get().runtime.output_dir),
            tag="sac_run"
        )

        # Save plot
        plot_path = Path(
            HydraConfig.get().runtime.output_dir) / "validate" / f"eval_plot_step{current_timestep}.png"
        plot_eval_curve(self.eval_envsteps, self.eval_means, self.eval_stds, plot_path)

    def update(
            self,
            obs_batch: torch.Tensor,
            act_batch: torch.Tensor,
            rew_batch: torch.Tensor,
            next_obs_batch: torch.Tensor,
            tm_batch: torch.Tensor,
    ):
        """
        Update function that updates critics, actor, and entropy coefficient.
        """

        # Pre-compute commonly used terms
        rew_batch = rew_batch.unsqueeze(-1)
        not_done = 1 - tm_batch.unsqueeze(-1).float()
        ent_coef = self.log_ent_coef.exp()

        # Compute target for critics
        with torch.no_grad():
            next_action, next_log_prob = self.actor_target(next_obs_batch)
            q1_target_val = self.q1_target(next_obs_batch, next_action)
            q2_target_val = self.q2_target(next_obs_batch, next_action)
            q_target_min = torch.min(q1_target_val, q2_target_val)

            # Target Q-value with entropy term
            q_target = rew_batch + self.gamma * not_done * (
                    q_target_min - ent_coef * next_log_prob.sum(dim=-1, keepdim=True))

        # Update both Q-functions
        for q_func, q_opt in [(self.q1, self.q1_optimizer), (self.q2, self.q2_optimizer)]:
            q_pred = q_func(obs_batch, act_batch)
            q_loss = F.mse_loss(q_pred, q_target)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()

        # Update actor
        new_action, new_log_prob = self.actor(obs_batch.float())
        q1_new, q2_new = self.q1(obs_batch, new_action), self.q2(obs_batch, new_action)
        min_q = torch.min(q1_new, q2_new)
        actor_loss = (-min_q + ent_coef * new_log_prob.sum(dim=-1, keepdim=True)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update entropy coefficient
        ent_coef_loss = -(self.log_ent_coef.exp() * (new_log_prob.detach() + self.target_entropy)).mean()

        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        # Polyak averaging for target networks
        for param, target_param in zip(
                [self.q1, self.q2, self.actor],
                [self.q1_target, self.q2_target, self.actor_target]
        ):
            polyak_update(param.parameters(), target_param.parameters(), self.tau)


