# import torch
import torch
# Dataloader import
from data import PascalVOCSegmentation
# Models
from segmodel import MobileNetV3_SegNet
from fcnresnet import FCNResNetWrapper
# Response based learning
import distill_response_loss as drl
import distill_features as df
from plotter import plot_history
from datetime import datetime
import os
from distill import load_model
import matplotlib.pyplot as plt
from train_segmodel import calculate_miou

if __name__ == "__main__":
    data = PascalVOCSegmentation(batch_size=1, n_workers=4)
    train_loader = data.get_train_loader()
    val_loader = data.get_train_loader()
    save_path = f"distill_{datetime.now().strftime("%Y%M%d_%H-%M-%S")}"
    os.mkdir(save_path)
    print(f"Created {save_path}")
    # model = load_model('./distill_20252808_18-28-31/final.pth')
    # model = load_model()
    # model = load_model('./distill_20255909_17-59-22/final.pth')
    # model = load_model('./20255909_20-59-29/final.pth')
    model = load_model('./best/f/final.pth')
    # model = FCNResNetWrapper()
    images, targets = next(iter(val_loader))
    model.to('cuda')
    model.eval()
    images = images.to('cuda')
    # pred is the raw output from the model, shape [B, C, H, W]
    pred = model(images)['out']  # example
    
    miou = calculate_miou(pred, targets.to('cuda'))
    print(f"mIoU: {miou}")

    print("Raw pred shape:", pred.shape)

    pred_classes = torch.argmax(pred, dim=1)
    print("After argmax:", pred_classes.shape)

    pred_classes = pred_classes.squeeze().cpu().numpy()
    print("After squeeze+numpy:", pred_classes.shape)

    # Get image and prediction
    img = images[0].cpu().permute(1, 2, 0).numpy()
    label = targets.cpu().permute(1, 2, 0).numpy()
    pred = pred_classes

    # Optional: unnormalize the image if it was normalized
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    img = (img * std + mean).clip(0, 1)

    # Display both
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(pred)
    axes[1].set_title("Predicted Segmentation")
    axes[1].axis("off")

    axes[2].imshow(label)
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.show()
