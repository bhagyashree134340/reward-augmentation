import numpy as np
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size=512):
        super().__init__()

        self.target = nn.Sequential(
            layer_init(nn.Linear(input_size, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, output_size))
        )

        self.predictor = nn.Sequential(
            layer_init(nn.Linear(input_size, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(512, output_size))
        )

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs):  # obs: [B, input_size]
        target_feature = self.target(obs).detach()
        predict_feature = self.predictor(obs)
        return predict_feature, target_feature


# class RNDModel(nn.Module):
#     def __init__(self, input_size, output_size=512):
#         super().__init__()
#
#         self.predictor = nn.Sequential(
#             nn.Linear(input_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_size)
#         )
#
#         self.target = nn.Sequential(
#             nn.Linear(input_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_size)
#         )
#
#         for param in self.target.parameters():
#             param.requires_grad = False
#
#     def forward(self, x):
#         pred = self.predictor(x)
#         target = self.target(x).detach()
#         return pred, target


