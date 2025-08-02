import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    def __init__(self, encoder: nn.Module, encoder_output_dim: int, action_dim: int, action_low: np.array, action_high: np.array):
        """
        Initialize the Actor network.

        :param obs_dim: dimention of the observations
        :param num_actions: dimention of the actions
        """
        super(Actor, self).__init__()
        # We are registering scale and bias as buffers so they can be saved and loaded as part of the model.
        # Buffers won't be passed to the optimizer for training!

        self.encoder = encoder

        self.register_buffer(
            "action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32)
        )

        self.fc_mu = nn.Linear(encoder_output_dim, action_dim)
        self.fc_std = nn.Linear(encoder_output_dim, action_dim)
        self.softplus = nn.Softplus()

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the actor network.

        return: mean_action, log_prob_action
        """
        x = self.encoder(obs)
        mu = self.fc_mu(x)

        log_std_min = -20
        log_std_max = 2
        log_std = self.fc_std(x)
        log_std = torch.clamp(log_std, min=log_std_min, max=log_std_max)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        tanh_action = torch.tanh(action)
        adjusted_action = tanh_action * self.action_scale + self.action_bias
        # adjusted_log_prob = log_prob - torch.log(...).sum(dim=-1, keepdim=True)

        log_det_jacobian = torch.log(self.action_scale * (1 - tanh_action.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
        adjusted_log_prob = log_prob - log_det_jacobian

        return adjusted_action, adjusted_log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int,
                 encoder: nn.Module = None,
                 encoder_output_dim: int = None):

        super().__init__()
        self.encoder = encoder
        self.use_encoder = encoder is not None

        input_dim = encoder_output_dim + action_dim if self.use_encoder else obs_dim + action_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.use_encoder:
            obs = self.encoder(obs)
        x = torch.cat([obs, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


