import torch
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import numpy as np
from dataloader import Coco2014Dataset

# This script assumes the Coco2014 and Coco2014DataModule classes
# are defined and available in the execution environment.


def denormalize_and_display_batch(images: torch.Tensor, labels: list, mean: tuple, std: tuple):
    """Denormalizes a batch of images and displays them using Matplotlib."""

    # Define mean and std tensors for denormalization
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)

    # Denormalize the batch: image * std + mean
    denorm_images = images * std_tensor + mean_tensor

    # Clamp values to [0, 1] as denormalization can overshoot
    denorm_images = torch.clamp(denorm_images, 0, 1)

    # Convert to NumPy array and rearrange dimensions for Matplotlib (C, H, W -> H, W, C)
    np_images = denorm_images.permute(0, 2, 3, 1).cpu().numpy()

    num_images = np_images.shape[0]

    # Set up the plot grid (e.g., 2x2 for batch size 4)
    rows = (num_images + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10, rows * 5))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    plt.suptitle("Batch of Sample Images and Labels", fontsize=16)

    for i in range(num_images):
        if i < len(labels):
            title = f"Label: {labels[i]}"
        else:
            title = "Label: N/A"

        axes[i].imshow(np_images[i])
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')  # Hide axes ticks and labels

    # Hide any unused subplots
    for j in range(num_images, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to make room for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    # --- Configuration ---
    COCO_ROOT = './coco2014'
    TEST_BATCH_SIZE = 4

    # 1. Define custom training augmentation
    # custom_train_augments = transforms.Compose([
    #     transforms.RandomResizedCrop(256),
    #     transforms.ColorJitter(brightness=0.1, contrast=0.1)
    # ])

    # 2. Instantiate the Data Module
    try:
        dm = Coco2014Dataset(
            root=COCO_ROOT,
            # Set to None to return text labels (not embeddings)
            train_encodings_path=None,
            batch_size=TEST_BATCH_SIZE,
            n_workers=0,
            pin=False
        )
    except NameError:
        print("ERROR: Coco2014DataModule is not defined. Ensure classes are available.")
        exit()

    # 3. Get the DataLoader (triggers lazy init on first access)
    train_loader = dm.train_dataloader()

    # 4. Grab the first batch
    start_time = time.time()
    try:
        images, labels = next(iter(train_loader))
    except Exception as e:
        print(f"Error fetching batch: {e}")
        exit()

    end_time = time.time()

    # 5. Display the Batch Information
    print("\n--- BATCH LOAD SUCCESS ---")
    print(f"Batch Size: {images.shape[0]}")
    print(f"Image Batch Shape: {images.shape}")

    if isinstance(labels, torch.Tensor):
        print(f"Labels Type: Tensor Embeddings, Shape: {labels.shape}")
        # Note: We cannot display embeddings, so we skip the visual plot here.
    else:
        print(f"Labels Type: Raw Text (List), Example: {labels[:2]}...")

        # --- 6. PLOT THE IMAGES ---
        # Get the mean/std from the data module setup (assuming train_ds is initialized now)
        try:
            display_mean = dm.train_ds.mean
            display_std = dm.train_ds.std
            denormalize_and_display_batch(
                images, labels, display_mean, display_std)
        except Exception as e:
            print(
                f"Could not display images. Make sure mean/std are accessible: {e}")

    print(f"Load Time: {end_time - start_time:.4f} seconds")
