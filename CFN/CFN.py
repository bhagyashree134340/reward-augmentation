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

        for param in self.prior.parameters():
            param.requires_grad = False

        # Running statistics to normalize prior output acc to the paper: E[‖f_prior(s)‖²] = 1
        self.register_buffer("prior_squared_norm_mean", torch.tensor(1.0))
        self.register_buffer("prior_squared_norm_count", torch.tensor(1e-4))

    def forward(self, state, update_prior_stats=False):
        """

        :param state:
        :param update_prior_stats:
        :return:
        """
        with torch.no_grad():
            prior_out = self.prior(state)

            if update_prior_stats:
                self.update_prior_stats(prior_out)

            norm_factor = (self.prior_squared_norm_mean + 1e-8).sqrt()
            normalized_prior = prior_out / norm_factor

        return self.net(state) + normalized_prior

    def compute_squared_output_norm(self, obs):
        """
        Compute the squared L2 norm (‖fϕ(s)‖²) of the network's output.
        """
        with torch.no_grad():
            output = self.forward(obs, update_prior_stats=False)
            return torch.norm(output, p=2, dim=-1) ** 2

    def update_prior_stats(self, prior_output):
        """
        Updates running estimate of E[‖f_prior(s)‖²]
        """
        with torch.no_grad():
            batch_squared_norms = prior_output.pow(2).sum(dim=-1)
            batch_mean = batch_squared_norms.mean()

            self.prior_squared_norm_count += 1
            delta = batch_mean - self.prior_squared_norm_mean
            self.prior_squared_norm_mean += delta / self.prior_squared_norm_count

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