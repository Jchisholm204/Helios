#!/usr/bin/env python3
"""
Test script for SnoutNet model
Tests the model with dummy input and verifies output shape
"""

import torch
import torch.nn as nn
from model import SnoutNet


def test_snoutnet():
    """Test SnoutNet model with dummy input"""
    
    print("=" * 50)
    print("Testing SnoutNet Model")
    print("=" * 50)
    
    # Initialize the model
    model = SnoutNet(input_channels=3, input_size=227)
    model.eval()  # Set to evaluation mode
    
    print(f"Model type: {model.type}")
    print(f"Input shape: {model.input_shape}")
    print()
    
    # Create dummy input tensor
    batch_size = 1 # Test with batch of 4 images
    dummy_input = torch.randn(batch_size, 3, 227, 227)
    print(f"Input tensor shape: {dummy_input.shape}")
    print(f"Expected output shape: ({batch_size}, 2)")
    print()
    
    # Test forward pass
    print("Running forward pass...")
    with torch.no_grad():  # Disable gradient computation for testing
        output = model(dummy_input)
    
    print(f"Output tensor shape: {output.shape}")
    print(f"Output tensor:\n{output}")
    print()
    
    # Verify output shape
    expected_shape = (batch_size, 2)
    if output.shape == expected_shape:
        print("✅ SUCCESS: Output shape matches expected (batch_size, 2)")
    else:
        print(f"❌ ERROR: Expected shape {expected_shape}, got {output.shape}")
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print layer-by-layer output shapes
    print(f"\nLayer-by-layer output shapes:")
    print(f"Input: {dummy_input.shape}")
    
    with torch.no_grad():
        x = dummy_input
        print(f"After Conv1: {model.conv1(x).shape}")
        x = torch.relu(model.bn1(model.conv1(x)))
        print(f"After Conv1 + ReLU: {x.shape}")
        
        print(f"After Conv2: {model.conv2(x).shape}")
        x = torch.relu(model.bn2(model.conv2(x)))
        print(f"After Conv2 + ReLU: {x.shape}")
        
        print(f"After Conv3: {model.conv3(x).shape}")
        x = torch.relu(model.bn3(model.conv3(x)))
        print(f"After Conv3 + ReLU: {x.shape}")
        
        # Flatten
        x_flat = x.view(x.size(0), -1)
        print(f"After flatten: {x_flat.shape}")
        
        # FC layers
        x = torch.relu(model.fc1(x_flat))
        print(f"After FC1 + ReLU: {x.shape}")
        x = model.dropout1(x)
        print(f"After dropout: {x.shape}")
        x = model.fc2(x)
        print(f"Final output: {x.shape}")


if __name__ == "__main__":
    test_snoutnet()
