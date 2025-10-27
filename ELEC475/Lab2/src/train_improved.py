#!/usr/bin/env python3
"""
Improved training script for SnoutNet model
Implements multiple optimizations for better coordinate regression performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from datetime import datetime
import math

from model import SnoutNet
from model_improved import ImprovedSnoutNet, SnoutNetWithSpatialAttention
from dataloader import SnoutDataset


class ImprovedSnoutNetTrainer:
    """Improved trainer class with multiple optimizations"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Multiple loss functions to try
        self.loss_functions = {
            'smooth_l1': nn.SmoothL1Loss(),
            'mse': nn.MSELoss(),
            'huber': nn.HuberLoss(delta=1.0),
            'l1': nn.L1Loss(),
            'wing_loss': self._wing_loss,
            'focal_l1': self._focal_l1_loss
        }
        
        # Start with Wing Loss (better for coordinate regression)
        self.criterion = self.loss_functions['wing_loss']
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
    def _wing_loss(self, pred, target, omega=10, epsilon=2):
        """Wing Loss - better for coordinate regression than MSE"""
        diff = torch.abs(pred - target)
        C = omega - omega * math.log(1 + omega / epsilon)
        
        loss = torch.where(
            diff < omega,
            omega * torch.log(1 + diff / epsilon),
            diff - C
        )
        return torch.mean(loss)
    
    def _focal_l1_loss(self, pred, target, alpha=2, gamma=2):
        """Focal L1 Loss - focuses on hard examples"""
        diff = torch.abs(pred - target)
        weight = alpha * torch.pow(diff, gamma)
        return torch.mean(weight * diff)
    
    def train_epoch(self, optimizer, scheduler=None):
        """Train for one epoch with improved techniques"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Learning rate scheduling
            if scheduler and hasattr(scheduler, 'step'):
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs, learning_rate=0.0005, weight_decay=1e-4, save_dir='./checkpoints'):
        """Train the model with improved configuration"""
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Improved optimizer configuration
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Improved learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,  # Peak learning rate
            epochs=num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,  # Warmup for 10% of training
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )
        
        print(f"Starting improved training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Loss function: Wing Loss")
        print(f"Optimizer: AdamW with OneCycleLR")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("-" * 60)
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(optimizer, scheduler)
            
            # Validate
            val_loss = self.validate()
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.epochs.append(epoch + 1)
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"{save_dir}/best_model.pth")
                print(f"  New best model saved! (Val Loss: {val_loss:.6f})")
            
            print("-" * 60)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        # Save final model
        self.save_model(f"{save_dir}/final_model.pth")
        
        return self.train_losses, self.val_losses
    
    def save_model(self, filepath):
        """Save model state dict"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs': self.epochs
        }, filepath)
    
    def plot_losses(self, save_path=None):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.epochs, self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Improved SnoutNet Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss plot saved to {save_path}")
        
        plt.show()


def create_improved_data_loaders(batch_size=64, num_workers=4, target_size=227):
    """Create training and validation data loaders with augmentation"""
    
    # Improved data transforms with augmentation
    from torchvision import transforms
    
    train_transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Training dataset with augmentation
    train_dataset = SnoutDataset(
        labels_file="oxford-iiit-pet-noses/train_noses.txt",
        img_dir="oxford-iiit-pet-noses/images-original/images",
        transform=train_transform,
        target_size=target_size,
        file_format='pet_noses'
    )
    
    # Validation dataset without augmentation
    val_dataset = SnoutDataset(
        labels_file="oxford-iiit-pet-noses/test_noses.txt",
        img_dir="oxford-iiit-pet-noses/images-original/images",
        transform=val_transform,
        target_size=target_size,
        file_format='pet_noses'
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # For consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader


def main():
    """Main training function with improved configuration"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Improved hyperparameters
    config = {
        'batch_size': 64,  # Increased batch size
        'num_epochs': 100,  # More epochs
        'learning_rate': 0.0005,  # Lower learning rate
        'weight_decay': 1e-4,
        'target_size': 227,
        'num_workers': 4,
        'model_type': 'improved'  # 'improved', 'attention', or 'original'
    }
    
    print("Improved Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create data loaders with augmentation
    print("Creating improved data loaders with augmentation...")
    train_loader, val_loader = create_improved_data_loaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        target_size=config['target_size']
    )
    
    # Create improved model
    print(f"Creating {config['model_type']} SnoutNet model...")
    if config['model_type'] == 'improved':
        model = ImprovedSnoutNet(input_channels=3, input_size=config['target_size'])
    elif config['model_type'] == 'attention':
        model = SnoutNetWithSpatialAttention(input_channels=3, input_size=config['target_size'])
    else:
        model = SnoutNet(input_channels=3, input_size=config['target_size'])
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model type: {model.type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Create improved trainer
    trainer = ImprovedSnoutNetTrainer(model, train_loader, val_loader, device)
    
    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./checkpoints/snoutnet_{config['model_type']}_{timestamp}"
    
    # Train the model
    print("Starting improved training...")
    train_losses, val_losses = trainer.train(
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        save_dir=save_dir
    )
    
    # Plot losses
    plot_path = f"{save_dir}/training_losses.png"
    trainer.plot_losses(save_path=plot_path)
    
    print(f"\nImproved training completed!")
    print(f"Model checkpoints saved to: {save_dir}")
    print(f"Loss plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
