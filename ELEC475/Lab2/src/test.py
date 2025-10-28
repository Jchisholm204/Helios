#!/usr/bin/env python3
"""
Test script for SnoutNet model
Calculates localization accuracy statistics on test dataset
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import DataLoader
import time

from model import SnoutNet
from model_improved import ImprovedSnoutNet, SnoutNetWithSpatialAttention
from dataloader import SnoutDataset


class SnoutNetTester:
    """Tester class for SnoutNet model evaluation"""
    
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
            model = ImprovedSnoutNet(input_channels=3, input_size=227).to(self.device)
            print("Detected: ImprovedSnoutNet")
        elif 'spatial_attention.0.weight' in state_dict:
            # SnoutNetWithSpatialAttention has spatial attention
            model = SnoutNetWithSpatialAttention(input_channels=3, input_size=227).to(self.device)
            print("Detected: SnoutNetWithSpatialAttention")
        else:
            # Original SnoutNet
            model = SnoutNet(input_channels=3, input_size=227).to(self.device)
            print("Detected: Original SnoutNet")
            
            # Initialize FC layers for original model
            dummy_input = torch.randn(1, 3, 227, 227).to(self.device)
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Load the state dict
        model.load_state_dict(state_dict)
        
        print(f"Model loaded successfully!")
        if 'train_losses' in checkpoint:
            print(f"Training history: {len(checkpoint['train_losses'])} epochs")
            if checkpoint['train_losses']:
                print(f"Final training loss: {checkpoint['train_losses'][-1]:.4f}")
                print(f"Final validation loss: {checkpoint['val_losses'][-1]:.4f}")
        
        return model
    
    def create_test_loader(self, batch_size=32, num_workers=4):
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
        
        print(f"Test dataset: {len(test_dataset)} samples")
        print(f"Test batches: {len(test_loader)}")
        
        return test_loader
    
    def calculate_euclidean_distance(self, pred_coords, true_coords):
        """Calculate Euclidean distance between predicted and true coordinates"""
        # pred_coords: [N, 2] - predicted (x, y) coordinates
        # true_coords: [N, 2] - ground truth (x, y) coordinates
        
        # Calculate Euclidean distance for each sample
        distances = torch.sqrt(torch.sum((pred_coords - true_coords) ** 2, dim=1))
        return distances
    
    def evaluate_model(self, test_loader, test_dataset):
        """Evaluate model on test dataset and calculate accuracy statistics"""
        print("Evaluating model on test dataset...")
        
        all_distances = []
        all_predictions = []
        all_ground_truth = []
        
        total_samples = 0
        inference_time = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_loader):
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                predictions = self.model(images)
                batch_time = time.time() - start_time
                inference_time += batch_time
                
                # Calculate distances with proper coordinate scaling
                batch_distances = []
                for i in range(len(images)):
                    pred_coord = predictions[i].cpu().numpy()
                    true_coord = targets[i].cpu().numpy()
                    
                    # Get original image size for scaling
                    sample_idx = batch_idx * test_loader.batch_size + i
                    if sample_idx < len(test_dataset):
                        filename = test_dataset.labels.iloc[sample_idx]['filename']
                        from PIL import Image
                        orig_img_path = f"oxford-iiit-pet-noses/images-original/images/{filename}"
                        orig_img = Image.open(orig_img_path)
                        orig_w, orig_h = orig_img.size
                        
                        # Scale ground truth coordinates to 227x227
                        scale_x = 227.0 / orig_w
                        scale_y = 227.0 / orig_h
                        true_coord_scaled = np.array([true_coord[0] * scale_x, true_coord[1] * scale_y])
                        
                        # Calculate distance
                        distance = np.sqrt((pred_coord[0] - true_coord_scaled[0])**2 + 
                                          (pred_coord[1] - true_coord_scaled[1])**2)
                        batch_distances.append(distance)
                
                # Store results
                all_distances.extend(batch_distances)
                all_predictions.extend(predictions.cpu().numpy())
                all_ground_truth.extend(targets.cpu().numpy())
                
                total_samples += len(images)
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx}/{len(test_loader)} batches ({total_samples} samples)")
        
        # Convert to numpy arrays
        all_distances = np.array(all_distances)
        all_predictions = np.array(all_predictions)
        all_ground_truth = np.array(all_ground_truth)
        
        # Calculate statistics
        stats = self._calculate_statistics(all_distances, inference_time, total_samples)
        
        return stats, all_distances, all_predictions, all_ground_truth
    
    def _calculate_statistics(self, distances, inference_time, total_samples):
        """Calculate localization accuracy statistics"""
        stats = {
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'median_distance': np.median(distances),
            'total_samples': total_samples,
            'inference_time': inference_time,
            'avg_inference_time': inference_time / total_samples,
            'samples_per_second': total_samples / inference_time if inference_time > 0 else 0
        }
        
        # Calculate percentiles
        stats['p25_distance'] = np.percentile(distances, 25)
        stats['p75_distance'] = np.percentile(distances, 75)
        stats['p90_distance'] = np.percentile(distances, 90)
        stats['p95_distance'] = np.percentile(distances, 95)
        
        return stats
    
    def print_statistics(self, stats):
        """Print formatted accuracy statistics"""
        print("\n" + "="*60)
        print("SNOUTNET LOCALIZATION ACCURACY STATISTICS")
        print("="*60)
        
        print(f"Total test samples: {stats['total_samples']}")
        print(f"Inference time: {stats['inference_time']:.2f} seconds")
        print(f"Average inference time: {stats['avg_inference_time']*1000:.2f} ms per sample")
        print(f"Throughput: {stats['samples_per_second']:.1f} samples/second")
        
        print(f"\nEuclidean Distance Statistics (pixels):")
        print(f"  Minimum:     {stats['min_distance']:.2f}")
        print(f"  Maximum:     {stats['max_distance']:.2f}")
        print(f"  Mean:        {stats['mean_distance']:.2f}")
        print(f"  Median:      {stats['median_distance']:.2f}")
        print(f"  Std Dev:     {stats['std_distance']:.2f}")
        
        print(f"\nPercentiles:")
        print(f"  25th:        {stats['p25_distance']:.2f}")
        print(f"  75th:        {stats['p75_distance']:.2f}")
        print(f"  90th:        {stats['p90_distance']:.2f}")
        print(f"  95th:        {stats['p95_distance']:.2f}")
        
        # Performance interpretation
        print(f"\nPerformance Interpretation:")
        if stats['mean_distance'] < 50:
            print("  🎯 EXCELLENT: Mean error < 50 pixels")
        elif stats['mean_distance'] < 100:
            print("  ✅ GOOD: Mean error < 100 pixels")
        elif stats['mean_distance'] < 200:
            print("  ⚠️  FAIR: Mean error < 200 pixels")
        else:
            print("  ❌ POOR: Mean error > 200 pixels")
        
        print("="*60)
    
    def plot_distance_distribution(self, distances, save_path=None):
        """Plot histogram of distance distribution"""
        plt.figure(figsize=(12, 8))
        
        # Main histogram
        plt.subplot(2, 2, 1)
        plt.hist(distances, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Euclidean Distance (pixels)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Localization Errors')
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 2)
        sorted_distances = np.sort(distances)
        cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
        plt.plot(sorted_distances, cumulative, 'r-', linewidth=2)
        plt.xlabel('Euclidean Distance (pixels)')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution of Errors')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 3)
        plt.boxplot(distances, vert=True)
        plt.ylabel('Euclidean Distance (pixels)')
        plt.title('Error Distribution Box Plot')
        plt.grid(True, alpha=0.3)
        
        # Statistics text
        plt.subplot(2, 2, 4)
        plt.axis('off')
        stats_text = f"""Statistics Summary:
        
Min: {np.min(distances):.2f} px
Max: {np.max(distances):.2f} px
Mean: {np.mean(distances):.2f} px
Median: {np.median(distances):.2f} px
Std: {np.std(distances):.2f} px

25th %ile: {np.percentile(distances, 25):.2f} px
75th %ile: {np.percentile(distances, 75):.2f} px
90th %ile: {np.percentile(distances, 90):.2f} px
95th %ile: {np.percentile(distances, 95):.2f} px

Total Samples: {len(distances)}"""
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distance distribution plot saved to {save_path}")
        
        plt.show()


def main():
    """Main testing function"""
    
    # Check if model path is provided
    if len(sys.argv) < 2:
        print("Usage: python test.py <model_path>")
        print("Example: python test.py ../checkpoints/snoutnet_20251026_183646/best_model.pth")
        return
    
    model_path = sys.argv[1]
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create tester
    tester = SnoutNetTester(model_path, device)
    
    # Create test loader
    test_loader = tester.create_test_loader(batch_size=32, num_workers=4)
    
    # Create test dataset for coordinate scaling
    test_dataset = SnoutDataset(
        labels_file="oxford-iiit-pet-noses/test_noses.txt",
        img_dir="oxford-iiit-pet-noses/images-original/images",
        target_size=227,
    )
    
    # Evaluate model
    stats, distances, predictions, ground_truth = tester.evaluate_model(test_loader, test_dataset)
    
    # Print statistics
    tester.print_statistics(stats)
    
    # Plot distance distribution
    plot_path = f"test_results_{os.path.basename(model_path).split('.')[0]}.png"
    tester.plot_distance_distribution(distances, save_path=plot_path)
    
    # Save detailed results
    results_path = f"test_results_{os.path.basename(model_path).split('.')[0]}.txt"
    with open(results_path, 'w') as f:
        f.write("SnoutNet Test Results\n")
        f.write("="*50 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Test samples: {stats['total_samples']}\n\n")
        
        f.write("Distance Statistics (pixels):\n")
        for key, value in stats.items():
            if 'distance' in key:
                f.write(f"{key}: {value:.4f}\n")
        
        f.write(f"\nPerformance:\n")
        f.write(f"Total inference time: {stats['inference_time']:.4f} seconds\n")
        f.write(f"Average inference time: {stats['avg_inference_time']*1000:.4f} ms\n")
        f.write(f"Throughput: {stats['samples_per_second']:.2f} samples/second\n")
    
    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    main()
