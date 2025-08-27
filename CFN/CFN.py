import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_init(m, bias=0.01):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, bias)
    return m

def linear_init(m, gain=1.0, bias=0.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, bias)
    return m


class CoinFlipNetworkCNN(nn.Module):
    """
    f(s) = f_hat(s) + norm(f_prior(s))
    - Predictive head f_hat is trainable.
    - Prior encoder/head are frozen; we keep running mean/var of f_prior.
    """
    def __init__(self, obs_shape, coin_dim, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c, h, w = obs_shape
        self.coin_flip_dim = int(coin_dim)

        def make_encoder():
            return nn.Sequential(
                conv_init(nn.Conv2d(c,   32, kernel_size=3, stride=1, padding=1)),
                nn.ReLU(),
                conv_init(nn.Conv2d(32,  64, kernel_size=3, stride=1, padding=1)),
                nn.ReLU(),
                nn.Flatten(),
            )

        self.net_encoder   = make_encoder()
        self.prior_encoder = make_encoder()

        # feature dim from encoder
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feat_dim = self.net_encoder(dummy).shape[1]
        self.feature_dim = int(feat_dim)

        # Heads: small output gains to keep magnitudes sane
        self.net_head = nn.Sequential(
            linear_init(nn.Linear(self.feature_dim, 256), gain=0.1, bias=0.0),
            nn.ReLU(),
            linear_init(nn.Linear(256, self.coin_flip_dim), gain=0.01, bias=0.0),
        )
        self.prior_head = nn.Sequential(
            linear_init(nn.Linear(self.feature_dim, 256), gain=0.1, bias=0.0),
            nn.ReLU(),
            linear_init(nn.Linear(256, self.coin_flip_dim), gain=0.01, bias=0.0),
        )

        # Freeze the prior path
        for p in list(self.prior_encoder.parameters()) + list(self.prior_head.parameters()):
            p.requires_grad = False

        # Running stats for prior whitening
        self.register_buffer("prior_mean",   torch.zeros(self.coin_flip_dim))
        self.register_buffer("prior_var",    torch.ones(self.coin_flip_dim))   # start with unit variance
        self.register_buffer("prior_count",  torch.tensor(1))             # slow early drift

        self.to(self.device)

    @torch.no_grad()
    def _update_prior_stats(self, prior_batch: torch.Tensor):
        """Welford-style update of running mean/var for f_prior."""
        b = prior_batch.shape[0]
        batch_mean = prior_batch.mean(dim=0)
        batch_var  = prior_batch.var(dim=0, unbiased=False).clamp_min(1e-8)

        total = self.prior_count + b
        delta = batch_mean - self.prior_mean
        new_mean = self.prior_mean + delta * (b / total)
        m_a = self.prior_var * self.prior_count
        m_b = batch_var * b
        m2  = m_a + m_b + (delta**2) * (self.prior_count * b / total)
        new_var = (m2 / total).clamp_min(1e-8)

        self.prior_mean.copy_(new_mean)
        self.prior_var.copy_(new_var)
        self.prior_count.copy_(total)

    def forward(self, obs: torch.Tensor, update_prior_stats: bool = True) -> torch.Tensor:
        """
        obs: float tensor in [0,1], shape [B,C,H,W]
        returns f(s) in R^d, shape [B,d]
        """
        x = obs
        net_feat   = self.net_encoder(x)
        prior_feat = self.prior_encoder(x)

        pred  = self.net_head(net_feat)         # trainable component f_hat(s)
        prior = self.prior_head(prior_feat)     # frozen random prior

        # 1) Normalize using *stored* stats (Alg. step: compute B(s_t))
        # Tiny floor keeps whitening stable if a variance dimension collapses.
        std = torch.sqrt(self.prior_var + 1e-8).clamp_min(1e-8)
        prior_white = (prior - self.prior_mean) / std

        # 2) Now update stats with current f_prior(s) (Alg. step: update μ, σ^2)
        if update_prior_stats:
            self._update_prior_stats(prior.detach())

        return pred + prior_white

    @torch.no_grad()
    def compute_squared_output_norm(self, obs: torch.Tensor) -> torch.Tensor:
        """Return ||f(s)||^2 for a batch; obs already on device and in [0,1]."""
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