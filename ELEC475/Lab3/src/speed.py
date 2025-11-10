import torch
import torch.nn as nn
from data import PascalVOCSegmentation
from fcnresnet import FCNResNetWrapper
from distill import load_model
from torchmetrics.segmentation import MeanIoU
from datetime import datetime
import os
import time

if __name__ == "__main__":
    # Setup data
    data = PascalVOCSegmentation(batch_size=1, n_workers=4)
    val_loader = data.get_train_loader()  # or get_val_loader() if available

    # Load model
    save_path = f"distill_{datetime.now().strftime('%Y%m%d_%H-%M-%S')}"
    os.mkdir(save_path)
    print(f"Created {save_path}")

    model = load_model('./best/f/best_miou.pth')
    # model = FCNResNetWrapper()
    model.to('cuda')
    model.eval()

    # Print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")

    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    # Timing and metrics
    total_images = 0
    total_loss = 0.0
    start_time = time.time()
    iou_metric = MeanIoU(num_classes=21).to('cuda')

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to('cuda')
            targets = targets.to('cuda')

            outputs = model(images)['out']  # forward pass

            # Compute loss
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * images.size(0)

            # Compute mIoU
            preds = torch.argmax(outputs, dim=1)

            # Replace ignored pixels with a valid class (0)
            targets_clean = targets.clone()
            targets_clean[targets_clean == 255] = 0

            iou_metric.update(preds, targets_clean)

            total_images += images.size(0)

    end_time = time.time()
    elapsed = end_time - start_time
    images_per_sec = total_images / elapsed

    avg_loss = total_loss / total_images
    avg_miou = iou_metric.compute().item()

    print(f"Processed {total_images} images in {elapsed:.2f} seconds")
    print(f"Throughput: {images_per_sec:.2f} images/sec")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Average mIoU: {avg_miou:.4f}")

