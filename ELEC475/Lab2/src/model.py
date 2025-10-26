

import torch
import torch.nn.functional as F
import torch.nn as nn


class SnoutNet(nn.Module):
    """
    SnoutNet: A CNN for pet nose centerpoint detection
    Architecture: 3 convolutional layers + 2 fully connected layers
    Input: 3-channel images (227x227x3)
    Output: 2D coordinates (x, y) representing nose centerpoint
    """

    def __init__(self, input_channels=3, input_size=227):
        super(SnoutNet, self).__init__()
        
        # Convolutional layers
        # 227x227x3 -> 57x57x64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 57x57x64 -> 15x15x128  
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=4, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 15x15x128 -> 4x4x256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=4, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Calculate the size after conv layers
        # 227x227x3 -> 57x57x64 -> 15x15x128 -> 4x4x256
        # Let's calculate this dynamically
        self.conv_output_size = None
        
        # Fully connected layers (will be initialized after first forward pass)
        self.fc1 = None
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = None
        
        self.type = 'SnoutNet'
        self.input_shape = (3, input_size, input_size)

    def forward(self, x):
        # Convolutional layers with ReLU activation and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))  # 227x227x3 -> 57x57x64
        x = F.relu(self.bn2(self.conv2(x)))  # 57x57x64 -> 15x15x128
        x = F.relu(self.bn3(self.conv3(x)))  # 15x15x128 -> 4x4x256
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Initialize FC layers on first forward pass
        if self.fc1 is None:
            conv_output_size = x.size(1)
            self.fc1 = nn.Linear(conv_output_size, 1024).to(x.device)
            self.fc2 = nn.Linear(1024, 2).to(x.device)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))  # 1024 neurons
        x = self.dropout1(x)
        x = self.fc2(x)  # Output: (x, y) coordinates
        
        return x
