from torch import nn


class RewardForwardFilter:
    def __init__(self, gamma):
        self.gamma = gamma
        self.rewems = None

    def update(self, reward, gamma=None):
        gamma = gamma if gamma is not None else self.gamma
        if self.rewems is None:
            self.rewems = reward
        else:
            self.rewems = self.rewems * gamma + reward
        return self.rewems


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size=512):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        self.target = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        pred = self.predictor(x)
        target = self.target(x).detach()
        return pred, target
