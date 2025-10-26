"""
Pet Nose Dataset Visualization Tool

This module provides functions to visualize the pet nose dataset with nose positions marked.
It's designed for testing and validating the dataloader functionality by displaying images
with their corresponding nose coordinates overlaid as red markers and crosshairs.

Usage:
    python imshow.py  # Shows sample images with nose positions marked
"""

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torch
from PIL import Image
import numpy as np
from dataloader import SnoutDataset


def show_pet_noses(dataset, num_samples=16, figsize=(12, 12)):
    """
    Display a grid of pet images with nose positions marked.
    
    Args:
        dataset: SnoutDataset instance
        num_samples: Number of samples to display (will be arranged in a grid)
        figsize: Figure size for matplotlib
    """
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten() if grid_size > 1 else [axes]
    
    # Get original image size for coordinate scaling
    original_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor()
    ])
    
    for i in range(min(num_samples, len(dataset))):
        # Get sample
        img_tensor, coords = dataset[i]
        
        # Get original image for display (without normalization)
        filename = dataset.labels.iloc[i]['filename']
        img_path = f"../oxford-iiit-pet-noses/images-original/images/{filename}"
        original_img = Image.open(img_path).convert('RGB')
        
        # Convert to numpy for display
        img_array = np.array(original_img)
        
        # Get original coordinates
        orig_x = dataset.labels.iloc[i]['x']
        orig_y = dataset.labels.iloc[i]['y']
        
        # Scale coordinates to match resized image
        orig_width, orig_height = original_img.size
        scale_x = 227 / orig_width
        scale_y = 227 / orig_height
        
        # Display image
        axes[i].imshow(img_array)
        axes[i].set_title(f"{filename}\nNose: ({orig_x}, {orig_y})", fontsize=8)
        
        # Mark nose position with a red circle
        axes[i].plot(orig_x, orig_y, 'ro', markersize=8, markeredgecolor='yellow', markeredgewidth=2)
        
        # Add crosshair for better visibility
        axes[i].axhline(y=orig_y, color='red', alpha=0.3, linewidth=1)
        axes[i].axvline(x=orig_x, color='red', alpha=0.3, linewidth=1)
        
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def show_pet_noses_batch(dataloader, batch_idx=0):
    """
    Display a batch of pet images with nose positions marked.
    
    Args:
        dataloader: DataLoader instance
        batch_idx: Which batch to display (default: first batch)
    """
    # Get the specified batch
    for i, (imgs, coords) in enumerate(dataloader):
        if i == batch_idx:
            batch_size = imgs.shape[0]
            grid_size = int(np.ceil(np.sqrt(batch_size)))
            
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
            axes = axes.flatten() if grid_size > 1 else [axes]
            
            for j in range(batch_size):
                # Denormalize image for display
                img = imgs[j]
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
                img = torch.clamp(img, 0, 1)
                
                # Convert to numpy and transpose for matplotlib
                img_np = img.permute(1, 2, 0).numpy()
                
                # Get coordinates
                x, y = coords[j][0].item(), coords[j][1].item()
                
                # Display image
                axes[j].imshow(img_np)
                axes[j].set_title(f"Sample {j+1}\nNose: ({x:.0f}, {y:.0f})", fontsize=10)
                
                # Mark nose position
                axes[j].plot(x, y, 'ro', markersize=10, markeredgecolor='yellow', markeredgewidth=2)
                axes[j].axhline(y=y, color='red', alpha=0.3, linewidth=1)
                axes[j].axvline(x=x, color='red', alpha=0.3, linewidth=1)
                axes[j].axis('off')
            
            # Hide unused subplots
            for j in range(batch_size, len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            plt.show()
            break



def main_pet_noses():
    """Main function to demonstrate pet nose visualization"""
    from torch.utils.data import DataLoader
    
    print("Loading pet nose dataset...")
    
    # Create dataset
    dataset = SnoutDataset(
        labels_file="../oxford-iiit-pet-noses/train_noses.txt",
        img_dir="../oxford-iiit-pet-noses/images-original/images",
        target_size=227,
        file_format='pet_noses'
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print("Showing first 16 samples with nose positions marked...")
    
    # Show first 16 samples
    show_pet_noses(dataset, num_samples=16)
    
    print("\nShowing first batch from dataloader...")
    
    # Show first batch
    show_pet_noses_batch(dataloader, batch_idx=0)


if __name__ == "__main__":
    # Uncomment the line below to run pet nose visualization
    main_pet_noses()
