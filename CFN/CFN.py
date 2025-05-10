import torch
import torch.nn as nn


class CoinFlipNetwork(nn.Module):
    def __init__(self, state_dim, coin_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, coin_dim)
        )

        self.prior = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, coin_dim)
        )

        # Initialize running statistics for prior normalization
        self.register_buffer("prior_mean", torch.zeros(coin_dim))
        self.register_buffer("prior_var", torch.ones(coin_dim))
        self.register_buffer("prior_count", torch.tensor(1e-4))  # to avoid div by 0

    def forward(self, state, update_prior_stats=True):
        """
        Args:
            state (torch.Tensor): input tensor of shape [batch_size, state_dim]
            update_prior_stats (bool): whether to update running stats (True during training)

        Returns:
            torch.Tensor: output of the CFN, shape [batch_size, coin_dim]
        """
        # Compute prior output without tracking gradients
        prior_out = self.prior(state).detach()

        # Update stats only if requested (e.g., during training)
        if update_prior_stats:
            self.update_prior_stats(prior_out)

        # Normalize prior using running mean and variance (using Welford's algorithm)
        normalized_prior = (prior_out - self.prior_mean) / ((self.prior_var / self.prior_count) + 1e-8).sqrt()

        # Add normalized prior to trainable network output
        return self.net(state) + normalized_prior

    def compute_output_norm(self, obs):
        """
        Compute the L2 norm (Euclidean norm) of the network's output fϕ(s).

        Args:
            state (torch.Tensor): A single state tensor of shape [state_dim]
                                  or a batch of states [batch_size, state_dim]

        Returns:
            torch.Tensor: The norm(s) of the output(s), shape [batch_size] or scalar
        """
        with torch.no_grad():
            output = self.forward(obs, update_prior_stats=False)
            norm = torch.norm(output, p=2, dim=-1)
            return norm

    def update_prior_stats(self, prior_output):
        """
        Update running mean and variance using Welford's algorithm.
        This version processes each sample in the batch individually.
        """

        # TODO: make faster by calculating for the entire batch
        with torch.no_grad():
            for sample in prior_output:
                self.prior_count += 1
                delta = sample - self.prior_mean
                self.prior_mean += delta / self.prior_count
                delta2 = sample - self.prior_mean
                self.prior_var += delta * delta2
