import math
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F

import copy
import numpy as np
import itertools

from hydra.core.hydra_config import HydraConfig

from CFN.CFN import CoinFlipNetwork
from CFN.cfn_buffer import CFNReplayBuffer
from CFN.priority_util import get_coin_flips, compute_intrinsic_reward, compute_cfn_priority
from networks.network import Critic, Actor
from replay_buffer.replay_buffer import ReplayBuffer
from utils.actor_io import save_actor
from utils.polyak import polyak_update
import logging

from utils.stats import EpisodeStats
from utils.validate import validate_from_checkpoint

log = logging.getLogger(__name__)


class SACAgent:
    def __init__(self,
                 env,
                 gamma=0.99,
                 lr=0.001,
                 batch_size=64,
                 tau=0.005,
                 maxlen=100_000,
                 target_entropy=-1.0,
                 cfn_cfg=None
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

        if cfn_cfg is not None:
            self.cfn_cfg = cfn_cfg
            self.coin_flip_dim = cfn_cfg.cfn_coin_flip_dim

            # initialize CFN and its optimizer
            self.cfn = CoinFlipNetwork(env.observation_space.shape[0], self.coin_flip_dim)
            self.cfn_optimizer = optim.Adam(self.cfn.parameters(), lr=self.cfn_cfg.cfn_lr)
            self.cfn_buffer = CFNReplayBuffer(max_size=self.cfn_cfg.cfn_replay_buffer_size)

        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.target_entropy = target_entropy

        # Initialize the Replay Buffer
        self.buffer = ReplayBuffer(maxlen)

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

    def train(self, num_episodes: int, max_steps: int) -> EpisodeStats:
        """
        Train the SAC agent.

        :param max_steps: max steps to capping episode length
        :param num_episodes: Number of episodes to train.
        :returns: The episode statistics.
        """
        # Keeps track of useful statistics
        stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
        )
        current_timestep = 0
        # TODO: wandb integration

        for i_episode in range(num_episodes):
            avg_reward = sum(stats.episode_rewards) / len(stats.episode_rewards)
            max_reward = max(stats.episode_rewards)
            min_reward = min(stats.episode_rewards)

            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 100 == 0:
                log.info(
                    f"Episode {i_episode + 1} of {num_episodes} | "
                    f"Time Step: {current_timestep} | "
                    f"Avg Rew: {avg_reward:.2f} | "
                    f"Max Rew: {max_reward:.2f} | "
                    f"Min Rew: {min_reward:.2f}"
                )
            # Reset the environment and get initial observation
            obs, _ = self.env.reset()

            for episode_time in itertools.count():
                # Choose action and execute
                with torch.no_grad():
                    action, _ = self.actor(torch.as_tensor(obs).float())
                    action = action.cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                if self.cfn_cfg is not None:
                    # compute intrinsic reward using eq 5
                    intrinsic_reward = compute_intrinsic_reward(self.coin_flip_dim,
                                                                self.cfn.compute_output_norm(torch.as_tensor(obs)
                                                                                             .float()))
                    reward += intrinsic_reward

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] += 1

                # Store sample in the replay buffer
                self.buffer.store(
                    torch.as_tensor(obs, dtype=torch.float32),
                    torch.as_tensor(action),
                    torch.as_tensor(reward, dtype=torch.float32),
                    # torch.as_tensor(intrinsic_reward, dtype=torch.float32),
                    torch.as_tensor(next_obs, dtype=torch.float32),
                    torch.as_tensor(terminated),
                )

                if self.cfn_cfg is not None:
                    # Sample random coin-flip vector
                    coin_flip = get_coin_flips(self.coin_flip_dim)

                    # add state coin-flip tuple to B_c
                    self.cfn_buffer.store(torch.tensor(obs), coin_flip.detach())

                # Sample a mini batch from the replay buffer
                obs_batch, act_batch, rew_batch, next_obs_batch, tm_batch = self.buffer.sample(self.batch_size)

                # Update for critics, actor, and entropy coefficient
                self.update(obs_batch, act_batch, rew_batch, next_obs_batch, tm_batch)

                if self.cfn_cfg is not None:
                    # sample minibatch from B_c and do an update
                    obs_batch_bc, coin_flip_batch_bc, sample_counts, indices = self.cfn_buffer.sample(self.batch_size)

                    self.update_cfn(obs_batch_bc, coin_flip_batch_bc)

                    # update priority for minibatch using eq 6
                    self.cfn_buffer.update_priorities(indices,
                                                      compute_cfn_priority(self.cfn, obs_batch_bc, sample_counts))

                current_timestep += 1

                # Check whether the episode is finished
                if terminated or truncated or episode_time >= max_steps:
                    break
                obs = next_obs

            if i_episode % 10 == 0:
                # Save actor
                # TODO: Add all paths to config maybe
                actor_path = Path(
                    HydraConfig.get().runtime.output_dir) / "checkpoints" / f"sac_actor_ep{i_episode:04d}.pt"
                actor_path.parent.mkdir(parents=True, exist_ok=True)
                save_actor(self.actor, actor_path)

                # Run validation
                validate_from_checkpoint(self.actor.__class__, self.env, actor_path, i_episode, max_steps)

        return stats

    def update(
            self,
            obs_batch: torch.Tensor,
            act_batch: torch.Tensor,
            rew_batch: torch.Tensor,
            next_obs_batch: torch.Tensor,
            tm_batch: torch.Tensor,
    ):
        """
         Update function that updates critics, actor, and entropy coefficient

        :param obs_batch: Batch of current observations.
        :param act_batch: Batch of actions.
        :param rew_batch: Batch of rewards.
        :param next_obs_batch: Batch of next observations.
        :param tm_batch: Batch of termination flags.
        """
        # Compute target for critics
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor_target(next_obs_batch)
            q1_target_value = self.q1_target(next_obs_batch, next_action)
            q2_target_value = self.q2_target(next_obs_batch, next_action)
            q_target_min = torch.min(q1_target_value, q2_target_value)

            # Compute target Q-value with entropy term
            q_target = rew_batch.unsqueeze(-1) + self.gamma * (1 - tm_batch.unsqueeze(-1).float()) * (
                    q_target_min - self.log_ent_coef.exp() * next_action_log_prob.sum(dim=-1, keepdim=True)
            ).mean(dim=-1, keepdim=True)

        # Update both q function using our target
        for q, optimizer in [(self.q1, self.q1_optimizer), (self.q2, self.q2_optimizer)]:
            q_pred = q(obs_batch, act_batch)
            q_loss = F.mse_loss(q_pred, q_target)

            optimizer.zero_grad()
            q_loss.backward()
            optimizer.step()

        # Update actor
        # Sample new actions and calculate log probabilities
        action, action_log_prob = self.actor(obs_batch.float())

        # Compute minimum Q-value for the new actions
        q1_new, q2_new = self.q1(obs_batch, action), self.q2(obs_batch, action)
        min_q = torch.min(q1_new, q2_new)

        # Calculate actor loss with entropy term
        entropy = -self.log_ent_coef.exp() * action_log_prob
        actor_loss = (-min_q - entropy).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update entropy coefficient
        ent_coef_loss = -(self.log_ent_coef.exp() * (action_log_prob + self.target_entropy).detach()).mean()
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()

        # Update target networks via Polyak averaging
        polyak_update(self.q1.parameters(), self.q1_target.parameters(), self.tau)
        polyak_update(self.q2.parameters(), self.q2_target.parameters(), self.tau)
        polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

    def update_cfn(self, obs_batch: torch.Tensor, coin_flip_batch: torch.Tensor):
        """
        Update the Coin Flip Network with the current batch of states.

        :param coin_flip_batch:
        :param obs_batch: Batch of current observations.
        """
        # Compute the predicted coin flip vectors for the batch of states
        predicted_coin_flips = self.cfn(obs_batch)

        # Compute the loss with respect to the coin flip vectors
        # (Assuming coin flip vectors are stored in the CFN replay buffer)

        # i think this will be from the buffer and the get_coin_flips_for_batch will be stored into the cfn buffer
        # coin_flips_batch = self.get_coin_flips_for_batch(obs_batch)

        cfn_loss = F.mse_loss(predicted_coin_flips, coin_flip_batch)

        # Backpropagate and update the CFN
        self.cfn_optimizer.zero_grad()
        cfn_loss.backward()
        self.cfn_optimizer.step()
