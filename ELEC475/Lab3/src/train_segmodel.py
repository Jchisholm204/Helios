# import torch
import torch
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data import PascalVOCSegmentation
from segmodel import MobileNetV3_SegNet
from tqdm import tqdm
from datetime import datetime
import json
import os

EPOCHS = 2000
LEARNING_RATE = 0.003
LEARNING_DECAY = 0.0003
BATCH_SIZE = 16
WORKERS = 8


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


def load_model(checkpoint_path: str = None, device='cuda'):
    print("Loading Model...")
    segnet = MobileNetV3_SegNet()
    if checkpoint_path:
        print(f"Loading Checkpoint {checkpoint_path}")
        segnet.load_state_dict(torch.load(checkpoint_path))
    segnet.to(device)
    n_train_params = sum(p.numel()
                         for p in segnet.parameters() if p.requires_grad)
    print(f"Loaded Model on {device} with {n_train_params} parameters")
    return segnet


def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, power=0.9):
    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calculate_miou(predictions, targets, num_classes=21):
    preds = predictions.argmax(dim=1)
    mask = (targets >= 0) & (targets < num_classes)
    preds = preds[mask]
    targets = targets[mask]

    intersection = torch.zeros(num_classes, device=predictions.device)
    union = torch.zeros(num_classes, device=predictions.device)

    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection[cls] = (pred_cls & target_cls).sum().float()
        union[cls] = (pred_cls | target_cls).sum().float()

    iou = intersection / (union + 1e-6)
    return torch.nanmean(iou).item()


def train_model(model, train_loader, val_loader, n_epochs, lr=1e-3, decay=1e-4,
                save_path="undefined", device='cuda'):
    # model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer_fn = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

    history = {
        'train_loss': [],
        'train_miou': [],
        'val_loss': [],
        'val_miou': []
    }

    best_iomu = 0
    best_loss = 99999

    for epoch in range(n_epochs):
        # --- Training Phase ---
        model.train()
        total_loss = 0.0
        total_miou = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {
                          epoch+1}/{n_epochs} [TRAIN]", leave=False)
        for images, targets in train_loop:
            images = images.to(device)
            targets = targets.to(device)

            optimizer_fn.zero_grad()
            outputs = model(images)['out']
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer_fn.step()

            total_loss += loss.item() * images.size(0)
            total_miou += calculate_miou(outputs.detach(), targets)

        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_train_miou = total_miou / len(train_loader)
        # First epoch has enormous error
        if epoch != 0:
            history['train_loss'].append(avg_train_loss)
            history['train_miou'].append(avg_train_miou)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {
                            epoch+1}/{n_epochs} [VAL]", leave=False)
            for images, targets in val_loop:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)['out']
                loss = loss_fn(outputs, targets)
                val_loss += loss.item() * images.size(0)
                val_miou += calculate_miou(outputs, targets)

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_miou = val_miou / len(val_loader)
        # First epoch has enormous error
        if epoch != 0:
            history['val_loss'].append(avg_val_loss)
            history['val_miou'].append(avg_val_miou)

        # Save Model if mIoU is the best so far
        if avg_val_miou > best_iomu:
            best_iomu = avg_val_miou
            torch.save(model.state_dict(), f'{save_path}/best_miou.pth')
            print(f"Saved Model with new best mIoU: {best_iomu:.4f}")
        # Save Model if loss is the best so far
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f'{save_path}/best_loss.pth')
            print(f"Saved Model with new best Val Loss: {best_iomu:.4f}")

        # --- LR Schedule ---
        poly_lr_scheduler(optimizer_fn, lr, epoch, n_epochs)

        print(
            f"Epoch [{epoch+1}/{n_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | Train mIoU: {
                avg_train_miou:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_miou:.4f}"
        )

    # Save Final Model
    torch.save(model.state_dict(), f'{save_path}/final.pth')
    # Save History
    with open(f'{save_path}/history.json', "w") as f:
        json.dump(history, f, indent=4)

    return history


if __name__ == "__main__":
    data = PascalVOCSegmentation(batch_size=BATCH_SIZE, n_workers=WORKERS)
    train_loader = data.get_train_loader()
    val_loader = data.get_train_loader()
    save_path = f"{datetime.now().strftime("%Y%M%d_%H-%M-%S")}"
    os.mkdir(save_path)
    print(f"Created {save_path}")
    model = load_model("./20255702_22-57-14/final.pth")
    history = train_model(model, train_loader, val_loader,
                          EPOCHS, LEARNING_RATE, LEARNING_DECAY,
                          save_path=save_path)
    plot_history(history, save_path=save_path)
