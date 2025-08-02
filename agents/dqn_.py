import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import minigrid
from cpprb import ReplayBuffer
import matplotlib.pyplot as plt
from collections import deque
import random

class DQN(nn.Module):
    """Deep Q-Network for MiniGrid environments"""
    
    def __init__(self, input_shape, n_actions, hidden_dim=256):
        super(DQN, self).__init__()
        
        # Calculate the size after conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Calculate conv output size
        conv_out_size = self._get_conv_out_size(input_shape)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def _get_conv_out_size(self, shape):
        """Calculate the output size of conv layers"""
        x = torch.zeros(1, *shape)
        x = self.conv(x)
        return int(np.prod(x.size()))
    
    def forward(self, x):
        """Forward pass"""
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

class DQNAgent:
    """DQN Agent with cpprb replay buffer"""
    
    def __init__(self, state_shape, n_actions, lr=3e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
                 buffer_size=50000, batch_size=32):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Networks
        self.q_network = DQN(state_shape, n_actions).to(self.device)
        self.target_network = DQN(state_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.update_target_network()
        
        # Training step counter
        self.training_steps = 0
        
        # Replay buffer using cpprb
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            {
                "obs": {"shape": state_shape, "dtype": np.float32},
                "act": {"shape": 1, "dtype": np.int32},
                "rew": {"shape": 1, "dtype": np.float32},
                "next_obs": {"shape": state_shape, "dtype": np.float32},
                "done": {"shape": 1, "dtype": np.bool_}
            }
        )
        
    def preprocess_state(self, state):
        """Preprocess MiniGrid observation"""
        # MiniGrid returns dict with 'image' key
        if isinstance(state, dict):
            state = state['image']
        
        # Convert to float and normalize
        state = state.astype(np.float32) / 255.0
        
        # Transpose to CHW format for PyTorch
        if len(state.shape) == 3:
            state = np.transpose(state, (2, 0, 1))
        
        return state
    
    def act(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        state = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        
        self.replay_buffer.add(
            obs=state,
            act=action,
            rew=reward,
            next_obs=next_state,
            done=done
        )
    
    def train(self):
        """Train the network on a batch of experiences"""
        if self.replay_buffer.get_stored_size() < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(batch["obs"]).to(self.device)
        actions = torch.LongTensor(batch["act"]).to(self.device)
        rewards = torch.FloatTensor(batch["rew"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_obs"]).to(self.device)
        dones = torch.BoolTensor(batch["done"]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_dqn(env_name="MiniGrid-Empty-5x5-v0", n_episodes=1000, 
              target_update_freq=100, save_freq=100):
    """Train DQN on MiniGrid environment"""
    
    # Create environment
    env = gym.make(env_name)
    
    # Get state and action dimensions
    obs = env.reset()[0]
    state_shape = (3, obs['image'].shape[0], obs['image'].shape[1])  # CHW format
    n_actions = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"State shape: {state_shape}")
    print(f"Number of actions: {n_actions}")
    
    # Create agent
    agent = DQNAgent(state_shape, n_actions)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    recent_rewards = deque(maxlen=100)
    
    for episode in range(n_episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Select action
            action = agent.act(state)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train()
            if loss is not None:
                losses.append(loss)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        recent_rewards.append(total_reward)
        
        # Print progress
        if episode % 50 == 0:
            avg_reward = np.mean(recent_rewards)
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Avg Loss: {avg_loss:.4f}")
        
        # Save model
        if episode % save_freq == 0 and episode > 0:
            torch.save(agent.q_network.state_dict(), f"dqn_model_{episode}.pth")
    
    env.close()
    
    return agent, episode_rewards, episode_lengths, losses

def plot_results(episode_rewards, episode_lengths, losses):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    
    # Moving average of rewards
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = [np.mean(episode_rewards[i:i+window]) 
                      for i in range(len(episode_rewards)-window+1)]
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
    
    # Episode lengths
    axes[1, 0].plot(episode_lengths)
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    
    # Training loss
    if losses:
        axes[1, 1].plot(losses)
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.show()

def test_agent(agent, env_name="MiniGrid-Empty-5x5-v0", n_episodes=10):
    """Test trained agent"""
    env = gym.make(env_name, render_mode="rgb_array")
    
    for episode in range(n_episodes):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state, training=False)  # No exploration
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            if done:
                print(f"Test Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")
                break
    
    env.close()

if __name__ == "__main__":
    # Train the agent
    print("Starting DQN training on MiniGrid...")
    agent, rewards, lengths, losses = train_dqn(
        env_name="MiniGrid-Empty-5x5-v0",
        n_episodes=1000,
        target_update_freq=100
    )
    
    # Plot results
    plot_results(rewards, lengths, losses)
    
    # Test the trained agent
    print("\nTesting trained agent...")
    test_agent(agent, "MiniGrid-Empty-5x5-v0", n_episodes=5)