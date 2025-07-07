import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
from collections import deque, namedtuple
import gymnasium as gym

import imageio
from pathlib import Path

import wandb

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


def evaluate_agent(agent, env_name="FrozenLake-v1", map_desc=None, max_timesteps=700, gif_path="eval.gif"):
    env = gym.make(env_name, is_slippery=False, render_mode="rgb_array", desc=map_desc, max_episode_steps=200)

    frames = []
    total_reward = 0
    state, _ = env.reset()
    done = False

    for t in range(max_timesteps):
        frame = env.render()
        frames.append(frame)

        state_tensor = torch.tensor([state], dtype=torch.long).to(agent.device)
        state_tensor = F.one_hot(state_tensor, num_classes=agent.state_size).float()

        with torch.no_grad():
            action = agent.qnetwork_local(state_tensor).argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        if done:
            frame = env.render()
            frames.append(frame)
            break

        state = next_state

    env.close()

    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, duration=0.3)
    print(f"Evaluation finished. Total Reward: {total_reward}")
    print(f"GIF saved at {gif_path}")


class DDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DDQN, self).__init__()

        self.seed = torch.manual_seed(SEED)
        self.input_shape = state_size
        self.action_size = action_size

        self.head_1 = nn.Linear(state_size, hidden_size)
        self.ff_1 = nn.Linear(hidden_size, hidden_size)
        self.ff_2 = nn.Linear(hidden_size, action_size)
        weight_init([self.head_1, self.ff_1])

    def forward(self, x):
        x = F.relu(self.head_1(x))
        x = F.relu(self.ff_1(x))
        return self.ff_2(x)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, state_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.state_size = state_size

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.tensor(np.vstack([e.state for e in experiences]), dtype=torch.long).to(self.device)
        states = F.one_hot(states.squeeze(-1), num_classes=self.state_size).float().to(self.device)
        actions = torch.tensor(np.vstack([e.action for e in experiences]), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.vstack([e.reward for e in experiences]), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.vstack([e.next_state for e in experiences]), dtype=torch.long).to(self.device)
        next_states = F.one_hot(next_states.squeeze(-1), num_classes=self.state_size).float().to(self.device)
        dones = torch.tensor(np.vstack([e.done for e in experiences]).astype(np.uint8), dtype=torch.float32).to(
            self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)


class MDQNAgent:
    def __init__(self, state_size, action_size, hidden_size, buffer_size, batch_size, gamma, tau, lr, update_every,
                 device):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.device = device
        self.batch_size = batch_size
        self.update_every = update_every
        self.step_count = 0

        self.qnetwork_local = DDQN(state_size, action_size, hidden_size).to(device)
        self.qnetwork_target = DDQN(state_size, action_size, hidden_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size, batch_size, device, state_size)

        self.entropy_tau = 0.03 #0.03
        self.alpha = 0.9 #0.9
        self.lo = -1

    def act(self, state, eps=0.):
        if random.random() > eps:
            state = torch.tensor([state], dtype=torch.long).to(self.device)
            state = F.one_hot(state, num_classes=self.state_size).float()
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done, t):
        self.memory.add(state, action, reward, next_state, done)

        self.step_count = (self.step_count + 1) % self.update_every
        if self.step_count == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            loss = self.learn(experiences, t)

    def learn(self, experiences, t):
        states, actions, rewards, next_states, dones = experiences

        q_targets_next = self.qnetwork_target(next_states).detach()
        logsum = torch.logsumexp((q_targets_next - q_targets_next.max(1)[0].unsqueeze(-1)) / self.entropy_tau,
                                 1).unsqueeze(-1)
        tau_log_pi_next = q_targets_next - q_targets_next.max(1)[0].unsqueeze(-1) - self.entropy_tau * logsum
        pi_target = F.softmax(q_targets_next / self.entropy_tau, dim=1)
        Q_target = (self.gamma * (pi_target * (q_targets_next - tau_log_pi_next) * (1 - dones)).sum(1)).unsqueeze(-1)

        q_k_targets = self.qnetwork_target(states).detach()
        v_k_target = q_k_targets.max(1)[0].unsqueeze(-1)
        logsum = torch.logsumexp((q_k_targets - v_k_target) / self.entropy_tau, 1).unsqueeze(-1)
        log_pi = q_k_targets - v_k_target - self.entropy_tau * logsum
        munchausen_addon = log_pi.gather(1, actions)
        munchausen_reward = rewards + self.alpha * torch.clamp(munchausen_addon, min=self.lo, max=0)

        Q_targets = munchausen_reward + Q_target

        Q_expected = self.qnetwork_local(states).gather(1, actions)

        with torch.no_grad():
            q_values = self.qnetwork_local(states)
            max_q_per_state, _ = q_values.max(dim=1, keepdim=True)
            action_gap_vector = max_q_per_state - q_values

            mean_gap = action_gap_vector.mean().item()
            max_gap = action_gap_vector.max().item()

            wandb.log({
                "gap/full_mean_gap": mean_gap,
            }, step=t)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        return loss.item()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


def train_agent(env_name="FrozenLake-v1", max_timesteps=50_000):
    wandb.init(
        project="frozenlake-cfn",
        name="MDQN-FrozenLake"
    )

    map = [
        "SHFFFFFF",
        "FFFFFHFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFHFFFFF",
        "FFFFFFFF",
        "FFFFFHFF",
        "FFFHFFFG",
    ]
    env = gym.make(env_name, is_slippery=False, render_mode="rgb_array", desc=map, max_episode_steps=200)

    env.reset(seed=SEED)
    env.action_space.seed(SEED)

    state_size = env.observation_space.n
    action_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = MDQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=256,
        buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        tau=0.005,
        lr=1e-3,
        update_every=1,
        device=device
    )

    eps = 1.0
    eps_decay = 0.99995
    eps_min = 0.01

    state, _ = env.reset()
    total_reward = 0
    episode = 1
    episode_lengths = []
    episode_rewards = []

    for t in range(1, max_timesteps + 1):
        action = agent.act(state, eps)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.step(state, action, reward, next_state, done, t)
        state = next_state
        total_reward += reward

        eps = max(eps_min, eps * eps_decay)

        if done:
            print(f"Episode {episode}\tTotal Reward: {total_reward:.1f}\tTimestep: {t}\tEpsilon: {eps:.3f}")
            episode_rewards.append(total_reward)
            episode_lengths.append(t)
            total_reward = 0
            state, _ = env.reset()
            episode += 1

    env.close()
    evaluate_agent(agent, env_name="FrozenLake-v1", map_desc=map, gif_path="frozenlake_eval.gif")


if __name__ == "__main__":
    train_agent()
