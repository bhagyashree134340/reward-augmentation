import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import copy
import numpy as np
from collections import namedtuple
import itertools

from network import Critic, Actor
from replay_buffer import ReplayBuffer
from utils import polyak_update

EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


class SACAgent:
    def __init__(self,
                 env,
                 gamma=0.99,
                 lr=0.001,
                 batch_size=64,
                 tau=0.005,
                 maxlen=100_000,
                 target_entropy=-1.0,
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

    def train(self, num_episodes: int) -> EpisodeStats:
        """
        Train the SAC agent.

        :param num_episodes: Number of episodes to train.
        :returns: The episode statistics.
        """
        # Keeps track of useful statistics
        stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
        )
        current_timestep = 0

        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 100 == 0:
                print(f'Episode {i_episode + 1} of {num_episodes}  Time Step: {current_timestep}')

            # Reset the environment and get initial observation
            obs, _ = self.env.reset()

            for episode_time in itertools.count():
                # Choose action and execute
                with torch.no_grad():
                    action, _ = self.actor(torch.as_tensor(obs).float())
                    action = action.cpu().numpy().clip(self.env.action_space.low, self.env.action_space.high)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] += 1

                # Store sample in the replay buffer
                self.buffer.store(
                    torch.as_tensor(obs, dtype=torch.float32),
                    torch.as_tensor(action),
                    torch.as_tensor(reward, dtype=torch.float32),
                    torch.as_tensor(next_obs, dtype=torch.float32),
                    torch.as_tensor(terminated),
                )

                # Sample a mini batch from the replay buffer
                obs_batch, act_batch, rew_batch, next_obs_batch, tm_batch = self.buffer.sample(self.batch_size)

                # Update the Critic network
                self.update_critics(
                    self.q1, self.q1_target, self.q1_optimizer,
                    self.q2, self.q2_target, self.q2_optimizer,
                    self.actor_target, self.log_ent_coef, self.gamma,
                    obs_batch, act_batch, rew_batch, next_obs_batch, tm_batch
                )

                # Update the Actor network
                self.update_actor(
                    self.q1,
                    self.q2,
                    self.actor,
                    self.actor_optimizer,
                    obs_batch.float(),
                    self.log_ent_coef,
                )

                # Update Entropy Coefficient
                self.update_entropy_coefficient(
                    self.actor,
                    self.log_ent_coef,
                    self.target_entropy,
                    self.ent_coef_optimizer,
                    obs_batch.float(),
                )

                # Update the target networks via Polyak Update
                polyak_update(self.q1.parameters(), self.q1_target.parameters(), self.tau)
                polyak_update(self.q2.parameters(), self.q2_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

                current_timestep += 1

                # Check whether the episode is finished
                if terminated or truncated or episode_time >= 500:
                    break
                obs = next_obs
        return stats

    def update_critics(
            self,
            q1: nn.Module,
            q1_target: nn.Module,
            q1_optimizer: optim.Optimizer,
            q2: nn.Module,
            q2_target: nn.Module,
            q2_optimizer: optim.Optimizer,
            actor_target: nn.Module,
            log_ent_coef: torch.Tensor,
            gamma: float,
            obs: torch.Tensor,
            act: torch.Tensor,
            rew: torch.Tensor,
            next_obs: torch.Tensor,
            tm: torch.Tensor,
    ):
        """
        Update both of SAC's critics for one optimizer step.

        :param log_ent_coef:
        :param q1: The first critic network.
        :param q1_target: The target first critic network.
        :param q1_optimizer: The first critic's optimizer.
        :param q2: The second critic network.
        :param q2_target: The target second critic network.
        :param q2_optimizer: The second critic's optimizer.
        :param actor: The actor network.
        :param actor_target: The target actor network.
        :param actor_optimizer: The actor's optimizer.
        :param gamma: The discount factor.
        :param obs: Batch of current observations.
        :param act: Batch of actions.
        :param rew: Batch of rewards.
        :param next_obs: Batch of next observations.
        :param tm: Batch of termination flags.

        """
        # 1. Calculate the target
        with torch.no_grad():
            next_action, next_action_log_prob = actor_target(next_obs)
            q1_target_value = q1_target(next_obs, next_action)
            q2_target_value = q2_target(next_obs, next_action)
            q_target_min = torch.min(q1_target_value, q2_target_value)

            q_target = rew.unsqueeze(-1) + gamma * (1 - tm.unsqueeze(-1).float()) * (
                    q_target_min - log_ent_coef.exp() * next_action_log_prob.sum(dim=-1, keepdim=True)).mean(dim=-1,
                                                                                                             keepdim=True)

        # 2. Update both q function using our target
        for q, optimizer in [(q1, q1_optimizer), (q2, q2_optimizer)]:
            q_pred = q(obs, act)
            q_loss = F.mse_loss(q_pred, q_target)

            optimizer.zero_grad()
            q_loss.backward()
            optimizer.step()

    def update_actor(
            self,
            q1: nn.Module,
            q2: nn.Module,
            actor: nn.Module,
            actor_optimizer: optim.Optimizer,
            obs: torch.Tensor,
            log_ent_coef: torch.Tensor,
    ):
        """
        Update the SAC's Actor network for one optimizer step.

        :param critic: The critic network.
        :param actor: The actor network.
        :param actor_optimizer: The actor's optimizer.
        :param obs: Batch of current observations.

        """
        # Actor Update
        action, action_log_prob = actor(obs)
        entropy = - log_ent_coef.exp() * action_log_prob
        q1, q2 = q1(obs, action), q2(obs, action)
        q1_q2 = torch.cat([q1, q2], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        actor_loss = (- min_q - entropy).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

    def update_entropy_coefficient(
            self,
            actor: nn.Module,
            log_ent_coef: torch.Tensor,
            target_entropy: float,
            ent_coef_optimizer: optim.Optimizer,
            obs: torch.Tensor,
    ):
        """
        Automatic update for entropy coefficient (alpha)

        :param actor: the actor network.
        :param log_ent_coef: tensor representing the log of entropy coefficient (log_alpha).
        :param target_entropy: tensor representing the desired target entropy.
        :param ent_coef_optimizer: torch optimizer for entropy coefficient.
        :param obs: current batch observation.
        """
        _, action_log_prob = actor(obs)
        ent_coef_loss = -(log_ent_coef.exp() * (action_log_prob + target_entropy).detach()).mean()
        ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        ent_coef_optimizer.step()
