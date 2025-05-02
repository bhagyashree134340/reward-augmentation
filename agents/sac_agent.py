import torch
import torch.optim as optim
import torch.nn.functional as F

import copy
import numpy as np
import itertools

from networks.network import Critic, Actor
from replay_buffer.replay_buffer import ReplayBuffer
from utils.polyak import polyak_update
import logging

from utils.validate import validate
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
        # TODO: log rewards every 100 eps then reset the accumulator
        # to accumulate rewards every 100 eps
        reward_accumulator = []

        for i_episode in range(num_episodes):
            # TODO: Also log the average return(?) per 100 eps maybe

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

                # Update for critics, actor, and entropy coefficient
                self.update(obs_batch, act_batch, rew_batch, next_obs_batch, tm_batch)

                current_timestep += 1

                # Check whether the episode is finished
                if terminated or truncated or episode_time >= max_steps:
                    break
                obs = next_obs

            if i_episode % 10 == 0:
                validate(self.actor, self.env, i_episode, max_steps)

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
