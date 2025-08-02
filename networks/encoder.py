import torch
import torch.nn as nn


class DDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, is_cnn=False):
        super(DDQN, self).__init__()
        if is_cnn:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            dummy_input = torch.zeros(1, 3, 7, 7)  
            encoded_size = self.encoder(dummy_input).shape[1]
            self.head = nn.Sequential(
                nn.Linear(encoded_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
            )
        self.is_cnn = is_cnn

    def forward(self, x):
        if self.is_cnn:
            x = x / 255.0  
            x = self.encoder(x)
            return self.head(x)
        else:
            return self.model(x)
