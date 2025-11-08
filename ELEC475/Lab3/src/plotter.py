import matplotlib.pyplot as plt


def plot_history(history, model_name="Model", save_path="."):
    """
    Generates and saves two separate plots: Loss and mIoU.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # --- Plot 1: Loss History ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'{model_name} Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)

    # --- Plot 2: mIoU History (Only Validation mIoU is collected reliably here) ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_miou'], 'r-', label='Validation mIoU')
    plt.plot(epochs, history['train_miou'], 'b-', label='Training mIoU')
    plt.title(f'{model_name} mIoU History')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plot_filename = f'{save_path}/{model_name}_training.png'
    plt.savefig(plot_filename)
    print(f"\nTraining plot saved to: {plot_filename}")
    plt.show()
    plt.close()
