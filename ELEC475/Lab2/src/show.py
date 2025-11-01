#!/usr/bin/env python3
"""
Visualization script for SnoutNet model predictions
Shows model predictions of nose locations on test images
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import sys
import random
from torch.utils.data import DataLoader
import argparse

from model import SnoutNet
from model_improved import ImprovedSnoutNet, SnoutNetWithSpatialAttention
from dataloader import SnoutDataset
from model_alexnet import AlexNetRegressor
from model_vgg import VGG16Regressor


class SnoutNetVisualizer:
    """Visualizer class for SnoutNet model predictions"""

    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path):
        """Load trained model from checkpoint"""
        print(f"Loading model from {model_path}")

        # Load checkpoint to detect model type
        checkpoint = torch.load(model_path, map_location=self.device)

        # Detect model type based on state dict keys
        state_dict = checkpoint['model_state_dict']
        if 'conv4.weight' in state_dict:
            # ImprovedSnoutNet has conv4 layer
            model = ImprovedSnoutNet(
                input_channels=3, input_size=227).to(self.device)
            print("Detected: ImprovedSnoutNet")
        elif 'spatial_attention.0.weight' in state_dict:
            # SnoutNetWithSpatialAttention has spatial attention
            model = SnoutNetWithSpatialAttention(
                input_channels=3, input_size=227).to(self.device)
            print("Detected: SnoutNetWithSpatialAttention")
        else:
            # Original SnoutNet
            # model = SnoutNet(input_channels=3, input_size=227).to(self.device)
            model = AlexNetRegressor().to(self.device)
            print("Detected: Original SnoutNet")

            # Initialize FC layers for original model
            dummy_input = torch.randn(1, 3, 227, 227).to(self.device)
            with torch.no_grad():
                _ = model(dummy_input)

        # Load the state dict
        model.load_state_dict(state_dict)

        print("Model loaded successfully!")
        return model

    def create_test_loader(self, batch_size=1, num_workers=0):
        """Create test data loader"""
        test_dataset = SnoutDataset(
            labels_file="oxford-iiit-pet-noses/test_noses.txt",
            img_dir="oxford-iiit-pet-noses/images-original/images",
            target_size=227,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return test_loader, test_dataset

    def denormalize_image(self, tensor_image):
        """Denormalize image tensor for display"""
        # ImageNet normalization values
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Denormalize
        denorm_image = tensor_image * std + mean
        denorm_image = torch.clamp(denorm_image, 0, 1)

        # Convert to numpy and transpose for matplotlib
        return denorm_image.permute(1, 2, 0).cpu().numpy()

    def calculate_distance(self, pred_coord, true_coord):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((pred_coord[0] - true_coord[0])**2 + (pred_coord[1] - true_coord[1])**2)

    def visualize_predictions(self, test_loader, test_dataset, num_samples=12, save_path=None):
        """Visualize model predictions on test images"""

        print(f"Generating predictions for {num_samples} test samples...")

        # Get random samples
        total_samples = len(test_dataset)
        if num_samples > total_samples:
            num_samples = total_samples

        # Select random indices
        sample_indices = random.sample(range(total_samples), num_samples)

        predictions = []
        ground_truths = []
        images = []
        distances = []

        with torch.no_grad():
            for idx in sample_indices:
                # Get single sample
                image, target = test_dataset[idx]
                image_batch = image.unsqueeze(0).to(self.device)
                target_batch = target.unsqueeze(0).to(self.device)

                # Get prediction
                prediction = self.model(image_batch)

                # Convert to numpy (coordinates are already normalized)
                pred_coord = prediction[0].cpu().numpy()
                true_coord = target.cpu().numpy()

                # Calculate distance using normalized coordinates (convert to pixels for display)
                distance = self.calculate_distance(pred_coord, true_coord) * 227

                # Store results
                predictions.append(pred_coord)
                ground_truths.append(true_coord)
                images.append(image)
                distances.append(distance)

        # Create visualization
        self._create_visualization_grid(images, predictions, ground_truths, distances,
                                        sample_indices, test_dataset, save_path)

    def _create_visualization_grid(self, images, predictions, ground_truths, distances,
                                   sample_indices, test_dataset, save_path):
        """Create grid visualization of predictions"""

        num_samples = len(images)
        cols = 4
        rows = (num_samples + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            row = i // cols
            col = i % cols
            ax = axes[row, col]

            # Denormalize image
            # image_np = self.denormalize_image(images[i])
            image_np = images[i].permute(1, 2, 0).cpu().numpy()

            # Display image
            ax.imshow(image_np)
            ax.set_title(
                f"Sample {i+1}\nError: {distances[i]:.1f}px", fontsize=10)

            # Plot ground truth (green circle) - convert normalized coords to pixel coords
            gt_x, gt_y = ground_truths[i]
            gt_x_pixel = gt_x * 227
            gt_y_pixel = gt_y * 227
            gt_circle = patches.Circle((gt_x_pixel, gt_y_pixel), 8, linewidth=2,
                                       edgecolor='green', facecolor='none', label='Ground Truth')
            ax.add_patch(gt_circle)

            # Plot prediction (red cross) - convert normalized coords to pixel coords
            pred_x, pred_y = predictions[i]
            pred_x_pixel = pred_x * 227
            pred_y_pixel = pred_y * 227
            ax.plot(pred_x_pixel, pred_y_pixel, 'r+', markersize=12,
                    markeredgewidth=3, label='Prediction')

            # Draw line connecting prediction to ground truth
            ax.plot([pred_x_pixel, gt_x_pixel], [pred_y_pixel, gt_y_pixel],
                    'b--', alpha=0.7, linewidth=1)

            ax.set_xlim(0, 227)
            ax.set_ylim(227, 0)  # Flip y-axis for image coordinates
            ax.set_xticks([])
            ax.set_yticks([])

            # Add legend only to first subplot
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)

        # Hide empty subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)

        plt.suptitle(f'SnoutNet Predictions on Test Images\n'
                     f'Green circles: Ground Truth, Red crosses: Predictions, Blue lines: Error vectors',
                     fontsize=14, y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()

        # Print summary statistics
        print(f"\nVisualization Summary:")
        print(f"Number of samples: {num_samples}")
        print(f"Mean error: {np.mean(distances):.2f} pixels")
        print(f"Min error: {np.min(distances):.2f} pixels")
        print(f"Max error: {np.max(distances):.2f} pixels")
        print(f"Std error: {np.std(distances):.2f} pixels")

    def visualize_single_prediction(self, test_dataset, sample_idx, save_path=None):
        """Visualize a single prediction in detail"""

        print(f"Visualizing sample {sample_idx}...")

        # Get sample
        image, target = test_dataset[sample_idx]
        image_batch = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(image_batch)

        # Convert to numpy (coordinates are already normalized)
        pred_coord = prediction[0].cpu().numpy()
        true_coord = target.cpu().numpy()

        distance = self.calculate_distance(pred_coord, true_coord) * 227

        # Create detailed visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Original image with predictions
        image_np = self.denormalize_image(image)
        ax1.imshow(image_np)

        # Ground truth - convert normalized coords to pixel coords
        gt_x_pixel = true_coord[0] * 227
        gt_y_pixel = true_coord[1] * 227
        gt_circle = patches.Circle((gt_x_pixel, gt_y_pixel), 10,
                                   linewidth=3, edgecolor='green', facecolor='none')
        ax1.add_patch(gt_circle)

        # Prediction - convert normalized coords to pixel coords
        pred_x_pixel = pred_coord[0] * 227
        pred_y_pixel = pred_coord[1] * 227
        ax1.plot(pred_x_pixel, pred_y_pixel, 'r+',
                 markersize=15, markeredgewidth=4)

        # Error line
        ax1.plot([pred_x_pixel, gt_x_pixel], [pred_y_pixel, gt_y_pixel],
                 'b--', alpha=0.8, linewidth=2)

        ax1.set_title(f'Sample {sample_idx}\nError: {distance:.2f} pixels')
        ax1.set_xlim(0, 227)
        ax1.set_ylim(227, 0)
        ax1.legend(['Prediction', 'Ground Truth', 'Error'], loc='upper right')

        # Zoomed view around the nose area - convert normalized coords to pixel coords
        margin = 50
        pred_x_pixel = pred_coord[0] * 227
        pred_y_pixel = pred_coord[1] * 227
        gt_x_pixel = true_coord[0] * 227
        gt_y_pixel = true_coord[1] * 227
        
        x_min = int(max(0, min(pred_x_pixel, gt_x_pixel) - margin))
        x_max = int(min(227, max(pred_x_pixel, gt_x_pixel) + margin))
        y_min = int(max(0, min(pred_y_pixel, gt_y_pixel) - margin))
        y_max = int(min(227, max(pred_y_pixel, gt_y_pixel) + margin))

        ax2.imshow(image_np[y_min:y_max, x_min:x_max])

        # Adjust coordinates for zoomed view
        gt_circle_zoom = patches.Circle((gt_x_pixel - x_min, gt_y_pixel - y_min), 8,
                                        linewidth=2, edgecolor='green', facecolor='none')
        ax2.add_patch(gt_circle_zoom)

        ax2.plot(pred_x_pixel - x_min, pred_y_pixel - y_min, 'r+', markersize=12, markeredgewidth=3)
        ax2.plot([pred_x_pixel - x_min, gt_x_pixel - x_min],
                 [pred_y_pixel - y_min, gt_y_pixel - y_min], 'b--', alpha=0.8, linewidth=2)

        ax2.set_title(f'Zoomed View\nError: {distance:.2f} pixels')
        ax2.set_xlim(0, x_max - x_min)
        ax2.set_ylim(y_max - y_min, 0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Single prediction visualization saved to {save_path}")

        plt.show()


def main():
    """Main visualization function"""

    parser = argparse.ArgumentParser(
        description='Visualize SnoutNet predictions')
    parser.add_argument('model_path', help='Path to trained model checkpoint')
    parser.add_argument('--num_samples', type=int, default=12,
                        help='Number of samples to visualize (default: 12)')
    parser.add_argument('--single', type=int, metavar='INDEX',
                        help='Visualize single sample at given index')
    parser.add_argument('--save', type=str, help='Save visualization to file')

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create visualizer
    visualizer = SnoutNetVisualizer(args.model_path, device)

    # Create test loader
    test_loader, test_dataset = visualizer.create_test_loader()

    print(f"Test dataset: {len(test_dataset)} samples")

    # Set random seed for reproducible results
    random.seed(42)

    if args.single is not None:
        # Visualize single sample
        if args.single >= len(test_dataset):
            print(f"Error: Sample index {
                  args.single} out of range (0-{len(test_dataset)-1})")
            return

        save_path = args.save or f"single_prediction_{args.single}.png"
        visualizer.visualize_single_prediction(
            test_dataset, args.single, save_path)
    else:
        # Visualize multiple samples
        save_path = args.save or f"predictions_grid_{
            os.path.basename(args.model_path).split('.')[0]}.png"
        visualizer.visualize_predictions(test_loader, test_dataset,
                                         args.num_samples, save_path)


if __name__ == "__main__":
    main()
