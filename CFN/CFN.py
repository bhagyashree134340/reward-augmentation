import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class CoinFlipNetworkCNN(nn.Module):
    def __init__(self, obs_shape, coin_dim, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c, h, w = obs_shape  # (3, 6, 6)

        # Compact CNN Encoder
        self.net_encoder = nn.Sequential(
            layer_init(nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1)),  # (6x6) → (6x6)
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),  # (6x6)
            nn.ReLU(),
            # nn.AdaptiveAvgPool2d((2, 2)),  # (6x6) → (2x2)
            nn.Flatten()  # 32×2×2 = 128
        )

        self.prior_encoder = nn.Sequential(
            layer_init(nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            # nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )

        self.feature_dim = 32 * 2 * 2  # = 128

        self.net_head = nn.Sequential(
            layer_init(nn.Linear(self.feature_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, coin_dim)),
        )

        self.prior_head = nn.Sequential(
            layer_init(nn.Linear(self.feature_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, coin_dim)),
        )

        # Freeze prior
        for param in self.prior_encoder.parameters():
            param.requires_grad = False
        for param in self.prior_head.parameters():
            param.requires_grad = False

        self.register_buffer("prior_mean", torch.zeros(coin_dim))
        self.register_buffer("prior_var", torch.ones(coin_dim) * 0.01)
        self.register_buffer("prior_count", torch.tensor(1.0))

        self.coin_flip_dim = coin_dim
        self.to(self.device)

    def forward(self, obs, update_prior_stats=False):
        obs = obs.to(self.device)
        if obs.dtype == torch.uint8 or torch.amax(obs) > 1.0:
            obs = obs.float() / 255.0

        net_features = self.net_encoder(obs)
        net_output = self.net_head(net_features)

        with torch.no_grad():
            prior_features = self.prior_encoder(obs)
            prior_out = self.prior_head(prior_features)

            if update_prior_stats:
                self.update_prior_stats(prior_out)

            std = torch.sqrt(self.prior_var + 1e-6)
            normalized_prior = (prior_out - self.prior_mean) / std
            normalized_prior = normalized_prior.detach()

        return net_output + normalized_prior

    def update_prior_stats(self, prior_out):
        with torch.no_grad():
            if prior_out.dim() > 1:
                batch_mean = prior_out.mean(dim=0)
            else:
                batch_mean = prior_out

            count = self.prior_count.item()
            new_count = count + 1

            delta = batch_mean - self.prior_mean
            self.prior_mean.add_(delta / new_count)

            delta2 = batch_mean - self.prior_mean
            self.prior_var.add_(delta * delta2 * count / new_count)
            self.prior_count.fill_(new_count)

    def compute_squared_output_norm(self, obs):
        obs = obs.to(self.device)
        with torch.no_grad():
            output = self.forward(obs, update_prior_stats=False)
            return torch.norm(output, p=2, dim=-1) ** 2


class CoinFlipNetwork(nn.Module):
    def __init__(self, state_dim, coin_dim, hidden_dim=128, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, coin_dim),
        )

        self.prior = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, coin_dim),
        )

        for param in self.prior.parameters():
            param.requires_grad = False

        self.register_buffer("prior_mean", torch.zeros(coin_dim))
        self.register_buffer("prior_var", torch.ones(coin_dim) * 0.01)
        self.register_buffer("prior_count", torch.tensor(1.0))

        self.coin_flip_dim = coin_dim
        self.to(self.device)

    def forward(self, state, update_prior_stats=False):
        state = state.to(self.device)

        net_output = self.net(state)

        with torch.no_grad():
            prior_out = self.prior(state)

            if update_prior_stats:
                self.update_prior_stats(prior_out)

            std = torch.sqrt(self.prior_var + 1e-6)
            normalized_prior = (prior_out - self.prior_mean) / std
            normalized_prior = normalized_prior.detach()

        return net_output + normalized_prior

    def update_prior_stats(self, prior_out):
        with torch.no_grad():
            if prior_out.dim() > 1:
                prior_out = prior_out.mean(dim=0)
                
            count = self.prior_count.item()
            new_count = count + 1

            delta = prior_out - self.prior_mean
            self.prior_mean.add_(delta / new_count)

            delta2 = prior_out - self.prior_mean
            self.prior_var.add_(delta * delta2 * count / new_count)
            self.prior_count.fill_(new_count)

    def compute_squared_output_norm(self, obs):
        obs = obs.to(self.device)
        with torch.no_grad():
            output = self.forward(obs, update_prior_stats=False)
            return torch.norm(output, p=2, dim=-1) ** 2