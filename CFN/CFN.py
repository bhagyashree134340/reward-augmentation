import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def layer_init(m, std=1.0):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(m.weight, gain=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    return m

class CoinFlipNetworkCNN(nn.Module):
    """
    cfn predictor + frozen random prior (paper-faithful)
      - conv stack keeps spatial resolution
      - prior is whitened with running mean/var
      - output f(s) = pred(s) + whitened_prior(s)
    """
    def __init__(self, obs_shape, coin_dim, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c, h, w = obs_shape  # e.g. (3, 6, 6)
        self.coin_flip_dim = int(coin_dim)

        # encoder (no pooling that collapses to 1x1/2x2)
        def make_encoder():
            return nn.Sequential(
                layer_init(nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
                nn.ReLU(),
                nn.Flatten(),
            )

        self.net_encoder   = make_encoder()
        self.prior_encoder = make_encoder()

        # infer feature dim safely
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feat_dim = self.net_encoder(dummy).shape[1]
        self.feature_dim = int(feat_dim)

        # heads
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

        # freeze prior
        for p in list(self.prior_encoder.parameters()) + list(self.prior_head.parameters()):
            p.requires_grad = False

        # running stats for prior whitening
        self.register_buffer("prior_mean",   torch.zeros(coin_dim))
        self.register_buffer("prior_var",    torch.ones(coin_dim) * 1e-2)  # small positive
        self.register_buffer("prior_count",  torch.tensor(1.0))

        self.to(self.device)

    @torch.no_grad()
    def _update_prior_stats(self, prior_batch):
        # welford-style update per dimension
        b = prior_batch.shape[0]
        batch_mean = prior_batch.mean(dim=0)
        batch_var  = prior_batch.var(dim=0, unbiased=False).clamp_min(1e-8)

        total = self.prior_count + b
        delta = batch_mean - self.prior_mean
        new_mean = self.prior_mean + delta * (b / total)
        # combine variances
        m_a = self.prior_var * self.prior_count
        m_b = batch_var * b
        m2  = m_a + m_b + (delta**2) * (self.prior_count * b / total)
        new_var = (m2 / total).clamp_min(1e-8)

        self.prior_mean.copy_(new_mean)
        self.prior_var.copy_(new_var)
        self.prior_count.copy_(total)

    def forward(self, obs, update_prior_stats=True):
        """
        obs: float tensor in [0,1], shape [B,C,H,W]
        returns: combined output f(s) in R^d (B,d)
        """
        x = obs
        net_feat   = self.net_encoder(x)
        prior_feat = self.prior_encoder(x)

        pred  = self.net_head(net_feat)             # (B,d)
        prior = self.prior_head(prior_feat)         # (B,d)

        if update_prior_stats:
            self._update_prior_stats(prior.detach())

        # whiten prior
        std = torch.sqrt(self.prior_var + 1e-8)
        prior_white = (prior - self.prior_mean) / std

        f = pred + prior_white
        return f

    @torch.no_grad()
    def compute_squared_output_norm(self, obs):
        """
        ||f(s)||^2 for B states (expects obs already on device and in [0,1])
        """
        f = self.forward(obs, update_prior_stats=False)  
        return (f.pow(2).sum(dim=1)).clamp_min(1e-12)



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