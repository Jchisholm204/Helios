#!/usr/bin/env python3
"""
Improved SnoutNet model architecture
Better suited for coordinate regression with improved spatial awareness
"""

import torch
import torch.nn.functional as F
import torch.nn as nn


class ImprovedSnoutNet(nn.Module):
    """
    Improved SnoutNet: Enhanced CNN for pet nose centerpoint detection
    Architecture improvements:
    - Smaller stride for better spatial resolution
    - Skip connections for better gradient flow
    - More appropriate receptive field
    - Better coordinate regression head
    """

    def __init__(self, input_channels=3, input_size=227):
        super(ImprovedSnoutNet, self).__init__()
        
        # Improved convolutional layers with smaller strides
        # 227x227x3 -> 114x114x64 (stride 2 instead of 4)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 114x114x64 -> 57x57x128 (stride 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 57x57x128 -> 29x29x256 (stride 2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 29x29x256 -> 15x15x512 (stride 2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Global average pooling instead of flattening
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Improved fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 2)  # Output: (x, y) coordinates
        
        # Initialize weights
        self._initialize_weights()
        
        self.type = 'ImprovedSnoutNet'
        self.input_shape = (3, input_size, input_size)

    def _initialize_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Convolutional layers with ReLU activation and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))  # 227x227x3 -> 114x114x64
        x = F.relu(self.bn2(self.conv2(x)))  # 114x114x64 -> 57x57x128
        x = F.relu(self.bn3(self.conv3(x)))  # 57x57x128 -> 29x29x256
        x = F.relu(self.bn4(self.conv4(x)))  # 29x29x256 -> 15x15x512
        
        # Global average pooling
        x = self.global_avg_pool(x)  # 15x15x512 -> 1x1x512
        x = x.view(x.size(0), -1)    # Flatten to 512
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))      # 512 -> 256
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))      # 256 -> 128
        x = self.dropout2(x)
        x = self.fc3(x)              # 128 -> 2 (x, y coordinates)
        
        return x


class SnoutNetWithSpatialAttention(nn.Module):
    """
    SnoutNet with spatial attention mechanism
    Focuses on relevant regions for nose detection
    """

    def __init__(self, input_channels=3, input_size=227):
        super(SnoutNetWithSpatialAttention, self).__init__()
        
        # Backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Coordinate regression head
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2)
        
        self.type = 'SnoutNetWithSpatialAttention'
        self.input_shape = (3, input_size, input_size)

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))  # 227x227x3 -> 114x114x64
        x = F.relu(self.bn2(self.conv2(x)))  # 114x114x64 -> 57x57x128
        x = F.relu(self.bn3(self.conv3(x)))  # 57x57x128 -> 29x29x256
        
        # Apply spatial attention
        attention = self.spatial_attention(x)
        x = x * attention
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Coordinate regression
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x


def create_model(model_type='improved', input_size=227):
    """Create model based on type"""
    if model_type == 'improved':
        return ImprovedSnoutNet(input_channels=3, input_size=input_size)
    elif model_type == 'attention':
        return SnoutNetWithSpatialAttention(input_channels=3, input_size=input_size)
    elif model_type == 'original':
        from model import SnoutNet
        return SnoutNet(input_channels=3, input_size=input_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the improved model
    print("Testing Improved SnoutNet...")
    model = ImprovedSnoutNet(input_channels=3, input_size=227)
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 227, 227)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Model type: {model.type}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test spatial attention model
    print("\nTesting SnoutNet with Spatial Attention...")
    model_att = SnoutNetWithSpatialAttention(input_channels=3, input_size=227)
    
    with torch.no_grad():
        output_att = model_att(dummy_input)
    
    print(f"Model type: {model_att.type}")
    print(f"Output shape: {output_att.shape}")
    
    total_params_att = sum(p.numel() for p in model_att.parameters())
    print(f"Total parameters: {total_params_att:,}")
