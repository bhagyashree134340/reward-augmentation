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
            
            # Match RND architecture - no aggressive pooling
            self.conv_layers = nn.Sequential(
                layer_init(nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)),  
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)), 
                nn.ReLU(),
                nn.Flatten(),  # Remove AdaptiveAvgPool2d for better spatial info
            )
            
            # Calculate feature size properly
            with torch.no_grad():
                dummy_input = torch.zeros(1, c, h, w)
                self.feature_size = self.conv_layers(dummy_input).shape[1]
            
            self.fc_layers = nn.Sequential(
                layer_init(nn.Linear(self.feature_size, hidden_size)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_size, action_size))
            )
            
        else:
            state_size = obs_shape
            self.fc_layers = nn.Sequential(
                layer_init(nn.Linear(state_size, hidden_size)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_size, hidden_size)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_size, action_size))
            )
        
        self.is_cnn = is_cnn

    def forward(self, x):
        if self.is_cnn:
            if x.dtype == torch.uint8:
                x = x.float() / 255.0
            elif x.max() > 1.0:
                x = x / 255.0
                
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)  
            return self.fc_layers(x)
        else:
            return self.fc_layers(x)