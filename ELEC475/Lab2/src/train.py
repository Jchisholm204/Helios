#!/usr/bin/env python3
"""
Training script for SnoutNet model
Trains the model on pet nose centerpoint detection using regression loss
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

from model import SnoutNet
from dataloader import SnoutDataset


class SnoutNetTrainer:
    """Trainer class for SnoutNet model"""

    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function for regression
        # MSE Loss - standard mean squared error loss
        # Works with normalized coordinates (0-1 range) from dataloader
        # Alternative: nn.SmoothL1Loss() for Huber loss, nn.L1Loss() for MAE
        self.criterion = nn.MSELoss()

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

    def train_epoch(self, optimizer):
        """Train for one epoch"""
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
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Print progress
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)
                                           }, Loss: {loss.item():.6f}')

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = images.to(
                    self.device), targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, num_epochs, learning_rate=0.001, weight_decay=1e-4, save_dir='./checkpoints'):
        """Train the model for specified number of epochs"""

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("-" * 60)

        best_val_loss = float('inf')
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(optimizer)

            # Validate
            val_loss = self.validate()

            # Update learning rate
            scheduler.step(val_loss)

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

    def load_model(self, filepath):
        """Load model state dict"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.epochs = checkpoint.get('epochs', [])

    def plot_losses(self, save_path=None):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses,
                 label='Training Loss', color='blue')
        plt.plot(self.epochs, self.val_losses,
                 label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('SnoutNet Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss plot saved to {save_path}")

        plt.show()


def create_data_loaders(batch_size=32, num_workers=4, target_size=227):
    """Create training and validation data loaders"""

    # Data transforms (no augmentation as specified)
    transform = None  # Will use default transform from SnoutDataset

    # Training dataset
    train_dataset = SnoutDataset(
        labels_file="oxford-iiit-pet-noses/train_noses.txt",
        img_dir="oxford-iiit-pet-noses/images-original/images",
        transform=transform,
        target_size=target_size,
    )

    # Validation dataset (using test partition)
    val_dataset = SnoutDataset(
        labels_file="oxford-iiit-pet-noses/test_noses.txt",
        img_dir="oxford-iiit-pet-noses/images-original/images",
        transform=transform,
        target_size=target_size,
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
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
    """Main training function"""

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    config = {
        'batch_size': 1,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-3,
        'target_size': 227,
        'num_workers': 8
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        target_size=config['target_size']
    )

    # Create model
    print("Creating SnoutNet model...")
    model = SnoutNet(input_channels=3, input_size=config['target_size'])

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # Create trainer
    trainer = SnoutNetTrainer(model, train_loader, val_loader, device)

    # Create timestamped save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./checkpoints/snoutnet_{timestamp}"

    # Train the model
    print("Starting training...")
    train_losses, val_losses = trainer.train(
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        save_dir=save_dir
    )

    # Plot losses
    plot_path = f"{save_dir}/training_losses.png"
    trainer.plot_losses(save_path=plot_path)

    print(f"\nTraining completed!")
    print(f"Model checkpoints saved to: {save_dir}")
    print(f"Loss plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
