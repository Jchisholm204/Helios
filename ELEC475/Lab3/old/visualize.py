import numpy as np
import matplotlib.pyplot as plt
# --- IMPORTANT: Denormalization Utility ---
# Your images were normalized using ImageNet MEAN and STD in 'transforms.py'.
# To display them correctly (in the 0-1 range), we must reverse this process.
# These values MUST match the ones used in transforms.py.
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    """Reverses the normalization process for image display."""
    img_np = tensor.cpu().numpy()

    if img_np.shape[0] == 1:  # single channel
        img_np = img_np.squeeze(0)  # [H,W]
        img_np = np.clip(img_np, 0, 1)
        return img_np  # Matplotlib can plot grayscale
    else:  # 3 channels
        mean = MEAN.reshape((3, 1, 1))
        std = STD.reshape((3, 1, 1))
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        return np.transpose(img_np, (1, 2, 0))


def visualize_sample_batch(data_loader, num_samples=4, title="Training Data Sample"):
    """
    Fetches a single batch from the data loader and displays a sample 
    of images and their corresponding segmentation masks.
    """

    # Set up the figure for display (2 rows for Image and Mask)
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    fig.suptitle(title, fontsize=16)

    # 1. Get the first batch
    try:
        images, targets = next(iter(data_loader))
    except StopIteration:
        print("Data loader is empty. Cannot retrieve a batch.")
        return

    # 2. Ensure we don't try to plot more samples than available in the batch
    plot_count = min(num_samples, images.size(0))

    # 3. Iterate over the sample images
    for i in range(plot_count):
        # --- Image Display (Top Row) ---
        img = images[i]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        img = denormalize(img)

        # Display the denormalized image
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Image {i+1}")
        axes[0, i].axis('off')

        # --- Mask Display (Bottom Row) ---
        # Move the mask tensor to CPU and convert to numpy array
        # The target mask contains integer class IDs (0 to 20 or 255 for ignore)
        mask = targets[i].cpu().numpy()
        # mask = mask.squeeze(0)

        # Display the mask. Use a 'jet' colormap for clear visualization
        # of class boundaries. 'interpolation="nearest"' prevents smoothing.
        axes[1, i].imshow(mask, cmap='jet', interpolation='nearest')
        axes[1, i].set_title(f"Mask {i+1}")
        axes[1, i].axis('off')

    # Adjust layout to prevent overlap and display the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

