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

    def forward(self, state, update_prior_stats=False):
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
        # normalized_prior = (prior_out - self.prior_mean) / ((self.prior_var / self.prior_count) + 1e-8).sqrt()
        normalized_prior = (prior_out - self.prior_mean) / (self.prior_var / self.prior_count + 1e-8).sqrt()

        # Add normalized prior to trainable network output
        return self.net(state) + normalized_prior

    def compute_squared_output_norm(self, obs):
        """
        Compute the squared L2 norm (‖fϕ(s)‖²) of the network's output.
        """
        # with torch.no_grad():
        #     output = self.forward(obs, update_prior_stats=False)
        #     return torch.sum(output ** 2)

        with torch.no_grad():
            output = self.forward(obs, update_prior_stats=False)
            norm = torch.norm(output, p=2, dim=-1)
            return norm ** 2

    def update_prior_stats(self, prior_output):
        with torch.no_grad():
            for i in range(prior_output.size(0)):
                value = prior_output[i]
                self.prior_count += 1
                delta = value - self.prior_mean
                self.prior_mean += delta / self.prior_count

                delta2 = value - self.prior_mean
                self.prior_var += delta * delta2

    # def update_prior_stats(self, prior_output):
    #     with torch.no_grad():
    #         batch_mean = prior_output.mean(dim=0)
    #         batch_var = prior_output.var(dim=0, unbiased=False)
    #         batch_size = prior_output.shape[0]
    #
    #         total_count = self.prior_count + batch_size
    #
    #         delta = batch_mean - self.prior_mean
    #         new_mean = self.prior_mean + delta * batch_size / total_count
    #
    #         m_a = self.prior_var * self.prior_count
    #         m_b = batch_var * batch_size
    #         M2 = m_a + m_b + delta ** 2 * self.prior_count * batch_size / total_count
    #         new_var = M2 / total_count
    #
    #         self.prior_mean = new_mean
    #         self.prior_var = new_var
    #         self.prior_count = total_count
