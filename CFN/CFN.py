import torch
import torch.nn as nn
import torch.nn.functional as F


class CoinFlipNetwork(nn.Module):
    def __init__(self, state_dim, coin_dim, hidden_dim=128, device=None):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, coin_dim),
            nn.Tanh()
        )

        self.prior = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, coin_dim),
            nn.Tanh()
        )

        for param in self.prior.parameters():
            param.requires_grad = False

        self.register_buffer("prior_mean", torch.zeros(coin_dim))
        self.register_buffer("prior_var", torch.ones(coin_dim))
        self.register_buffer("prior_count", torch.tensor(1))

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.coin_flip_dim = coin_dim

    def forward(self, state, update_prior_stats=False):
        state = state.to(self.device)

        with torch.no_grad():
            prior_out = self.prior(state)

            if update_prior_stats:
                self.update_prior_stats(prior_out)

            epsilon = 1e-4 
            std = torch.sqrt(self.prior_var + epsilon)
            normalized_prior = (prior_out - self.prior_mean) / std

            normalized_prior = normalized_prior.detach()

        return self.net(state) + normalized_prior


    def update_prior_stats(self, prior_out):
        with torch.no_grad():
            x = prior_out.squeeze(0)  
            count = self.prior_count.item()
            new_count = count + 1

            delta = x - self.prior_mean
            self.prior_mean.add_(delta / new_count)

            delta2 = x - self.prior_mean  
            self.prior_var.add_(delta * delta2 * count / new_count)

            self.prior_count.fill_(new_count)

    def compute_squared_output_norm(self, obs):
        obs = obs.to(self.device)
        with torch.no_grad():
            output = self.forward(obs, update_prior_stats=False)
            return torch.norm(output, p=2, dim=-1) ** 2

    def update_cfn_network(self, cfn, cfn_optimizer, obs_batch, coin_flip_batch):
        predicted_coin_flips = cfn(obs_batch)
        cfn_loss = F.mse_loss(predicted_coin_flips, coin_flip_batch)
        cfn_optimizer.zero_grad()
        cfn_loss.backward()
        cfn_optimizer.step()
