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

    def forward(self, state):
        return self.net(state)

    def compute_output_norm(self, state):
        """
        Compute the L2 norm (Euclidean norm) of the network's output fϕ(s).

        Args:
            state (torch.Tensor): A single state tensor of shape [state_dim]
                                  or a batch of states [batch_size, state_dim]

        Returns:
            torch.Tensor: The norm(s) of the output(s), shape [batch_size] or scalar
        """
        with torch.no_grad():
            output = self.forward(state)
            norm = torch.norm(output, p=2, dim=-1)
            return norm



