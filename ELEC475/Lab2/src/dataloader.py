import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import re

class SnoutDataset(Dataset):
    """
    Custom dataset for pet nose centerpoint detection.
    Supports both CSV format and pet noses text format.
    For pet noses format: expects text file with lines like 'filename.jpg,"(x, y)"'
    For CSV format: expects CSV file with columns: filename, x, y
    """

    def __init__(self, labels_file, img_dir, transform=None, target_size=227, file_format='auto'):
        self.img_dir = img_dir
        self.transform = transform
        self.target_size = target_size
        self.file_format = file_format
        
        # Parse labels based on file format
        self.labels = self._parse_labels_file(labels_file)

        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
    
    def _parse_labels_file(self, labels_file):
        """Parse labels file based on format detection or specified format."""
        if self.file_format == 'auto':
            # Auto-detect format based on file extension
            if labels_file.endswith('.csv'):
                return self._parse_csv_format(labels_file)
            else:
                return self._parse_pet_noses_format(labels_file)
        elif self.file_format == 'csv':
            return self._parse_csv_format(labels_file)
        elif self.file_format == 'pet_noses':
            return self._parse_pet_noses_format(labels_file)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
    
    def _parse_csv_format(self, csv_file):
        """Parse CSV format with columns: filename, x, y"""
        return pd.read_csv(csv_file)
    
    def _parse_pet_noses_format(self, txt_file):
        """Parse pet noses text format: filename.jpg,"(x, y)" """
        labels = []
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split by comma to separate filename and coordinates
                parts = line.split(',', 1)
                if len(parts) != 2:
                    continue
                
                filename = parts[0].strip()
                coords_str = parts[1].strip()
                
                # Extract x, y coordinates from "(x, y)" format
                match = re.match(r'"\((\d+),\s*(\d+)\)"', coords_str)
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    labels.append({'filename': filename, 'x': x, 'y': y})
        
        return pd.DataFrame(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load image
        filename = self.labels.iloc[idx]['filename']
        img_name = os.path.join(self.img_dir, filename)
        image = Image.open(img_name).convert('RGB')

        # Load label (x, y)
        x = self.labels.iloc[idx]['x']
        y = self.labels.iloc[idx]['y']
        label = torch.tensor([x, y], dtype=torch.float32)

        # Apply transforms
        image = self.transform(image)

        return image, label


# --- Example usage ---
if __name__ == "__main__":
    # Example with pet noses dataset
    dataset = SnoutDataset(
        labels_file="../oxford-iiit-pet-noses/train_noses.txt",
        img_dir="../oxford-iiit-pet-noses/images-original/images",
        target_size=227,
        file_format='pet_noses'
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Check one batch
    for batch_idx, (imgs, coords) in enumerate(dataloader):
        print("Batch of images:", imgs.shape)   # [B, 3, 227, 227]
        print("Batch of labels:", coords.shape) # [B, 2]
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
