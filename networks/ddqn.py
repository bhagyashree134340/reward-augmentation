import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class DDQN(nn.Module):
    def __init__(self, obs_shape, action_size, hidden_size, is_cnn=False):
        super(DDQN, self).__init__()
        
        if is_cnn:
            c, h, w = obs_shape
            
            # MiniGrid-specific CNN architecture
            self.conv_layers = nn.Sequential(
                layer_init(nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)),  # Same size
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),  # Same size
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),  # Same size
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),  # Reduce to 2x2
            )
            
            # Feature size is always 64 * 2 * 2 = 256 for MiniGrid
            self.feature_size = 64 * 2 * 2
            
            # Fully connected layers
            self.fc_layers = nn.Sequential(
                layer_init(nn.Linear(self.feature_size, hidden_size)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_size, action_size))
            )
            
        else:
            # For non-CNN case (if needed)
            state_size = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
            self.fc_layers = nn.Sequential(
                nn.Linear(state_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size)
            )
        
        self.is_cnn = is_cnn

    def forward(self, x):
        if self.is_cnn:
            # Normalize pixel values to [0, 1] if they're in [0, 255]
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            elif x.max() > 1.0:
                x = x / 255.0
                
            # Apply conv layers
            x = self.conv_layers(x)
            # Flatten
            x = x.view(x.size(0), -1)
            # Apply fully connected layers
            return self.fc_layers(x)
        else:
            return self.fc_layers(x)