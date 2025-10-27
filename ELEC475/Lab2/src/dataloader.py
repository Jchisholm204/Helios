#!/usr/bin/env python
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas
import os
import re


class SnoutDataset(Dataset):
    """
    Custom dataset for pet nose centerpoint detection.
    Supports both CSV format and pet noses text format.
    For pet noses format: expects text file with lines like 'filename.jpg,"(x, y)"'
    For CSV format: expects CSV file with columns: filename, x, y
    """

    def __init__(self, labels_file, img_dir, transform=None, target_size=227):
        self.img_dir = img_dir
        self.transform = transform
        self.target_size = target_size

        # Parse labels based on file format
        self.labels = self._parse_labels_file(labels_file)

    def _parse_labels_file(self, labels_file):
        labels = []
        with open(labels_file) as f:
            for line in f:
                if not line:
                    continue
                parts = line.split(',', 1)
                if len(parts) != 2:
                    continue
                filename = parts[0]
                coords_str = parts[1].strip()
                # Extract x, y coordinates from "(x, y)" format
                # match = re.match(r'"\((\d+),\s*(\d+)\)"', coords_str)
                match = re.match(r'"?\(\s*(\d+)\s*,\s*(\d+)\s*\)"?', coords_str)
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    labels.append({'filename': filename, 'x': x, 'y': y})

        print("Labels Size: ", len(labels))
        return pandas.DataFrame(labels)

    def __len__(self):
        return len(self.labels)

    # def __getitem__(self, idx):
    #     # Load image
    #     filename = self.labels.iloc[idx]['filename']
    #     img_path = os.path.join(self.img_dir, filename)
    #     image = Image.open(img_path).convert('RGB')
    #
    #     # Store original size before resizing
    #     orig_w, orig_h = image.size
    #
    #     # Load label (x, y) and normalize to [0, 1]
    #     x = self.labels.iloc[idx]['x'] / orig_w
    #     y = self.labels.iloc[idx]['y'] / orig_h
    #     label = torch.tensor([x, y], dtype=torch.float32)
    #
    #     # Apply base resize transform (normalize all to target size)
    #     base_transform = transforms.Resize(self.target_size)
    #     image = base_transform(image)
    #
    #     # Apply optional user transform (augmentation, etc.)
    #     if self.transform:
    #         image = self.transform(image)
    #
    #     # Convert to tensor if not done in transform
    #     if not isinstance(image, torch.Tensor):
    #         image = transforms.ToTensor()(image)
    #
    #     return image, label

    def __getitem__(self, idx):
        # Load image
        filename = self.labels.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        # Store original size before resizing
        orig_w, orig_h = image.size

        # Normalize label coordinates
        x = self.labels.iloc[idx]['x'] / orig_w
        y = self.labels.iloc[idx]['y'] / orig_h
        label = torch.tensor([x, y], dtype=torch.float32)

        # ✅ Always resize to the same (H, W)
        base_transform = transforms.Resize((self.target_size, self.target_size))
        image = base_transform(image)

        # Apply optional user-defined transform (e.g. augmentation)
        if self.transform:
            image = self.transform(image)

        # Ensure output is a tensor
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)

        return image, label



# --- Example usage ---
if __name__ == "__main__":
    print("Running Dataloader Test")
    # Example with pet noses dataset
    dataset = SnoutDataset(
        labels_file="oxford-iiit-pet-noses/train_noses.txt",
        img_dir="oxford-iiit-pet-noses/images-original/images",
        target_size=227,
    )
    for i in range(10):
        img, lbl = dataset[i]
        print(i, img.shape)
    exit(0)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Check one batch
    for batch_idx, (imgs, coords) in enumerate(dataloader):
        print("Batch of images:", imgs.shape)    # [B, 3, 227, 227]
        print("Batch of labels:", coords.shape)  # [B, 2]
        print("\nFirst batch details:")
        print("-" * 50)

        # Get the filenames for this batch
        start_idx = batch_idx * dataloader.batch_size
        end_idx = min(start_idx + dataloader.batch_size, len(dataset))

        for i in range(len(imgs)):
            sample_idx = start_idx + i
            if sample_idx < len(dataset.labels):
                filename = dataset.labels.iloc[sample_idx]['filename']
                x, y = coords[i][0].item(), coords[i][1].item()
                print(f"Sample {i+1:2d}: {filename:30s} -> ({x:3.0f}, {y:3.0f})")

        break
