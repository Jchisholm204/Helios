import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import SnoutNet
from dataloader import SnoutDataset


# --------------------------------------------------------------------
# Configuration (edit these paths & hyperparameters as needed)
# --------------------------------------------------------------------
train_data_dir = "oxford-iiit-pet-noses/images-original/images"
train_labels_file = "oxford-iiit-pet-noses/train_noses.txt"

val_data_dir = "oxford-iiit-pet-noses/images-original/images"
val_labels_file = "oxford-iiit-pet-noses/test_noses.txt"

save_dir = "./checkpoints"
num_epochs = 50
batch_size = 32
learning_rate = 1e-3
num_workers = 8
# --------------------------------------------------------------------


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for imgs, coords in tqdm(dataloader, desc="Training", leave=False):
        imgs, coords = imgs.to(device), coords.to(device)

        # Forward
        outputs = model(imgs)
        loss = criterion(outputs, coords)

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for imgs, coords in tqdm(dataloader, desc="Validating", leave=False):
            imgs, coords = imgs.to(device), coords.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, coords)
            running_loss += loss.item() * imgs.size(0)

    return running_loss / len(dataloader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # ----------------------------------------------------------------
    # Datasets and Dataloaders
    # ----------------------------------------------------------------
    train_dataset = SnoutDataset(
        img_dir=train_data_dir, labels_file=train_labels_file)
    val_dataset = SnoutDataset(
        img_dir=val_data_dir, labels_file=val_labels_file)
    print(f'Train Dataset Shape {train_dataset[0][0].shape}')
    print(f'Validation Dataset Shape {val_dataset[0][0].shape}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # ----------------------------------------------------------------
    # Model, Loss, Optimizer
    # ----------------------------------------------------------------
    model = SnoutNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Training Loop
    # ----------------------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")

        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, f"snoutnet_best.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Saved new best model to {checkpoint_path}")


if __name__ == "__main__":
    main()

