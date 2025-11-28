import argparse
import torch
import torch.optim as optim
import time
import os
from typing import Dict, Any
from tqdm import tqdm

# Assuming all these imports are available in the 'src' environment
from experiment_logger import ExperimentLogger
from core.model import CLIPModel
from core.loss import InfoNCELoss
from core.metrics import calculate_recall_at_k
from dataloader import Coco2014Dataset


# Define the main training and validation function
def train_and_validate(hparams: Dict[str, Any]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    logger = ExperimentLogger(experiment_name=hparams['exp_name'])
    logger.save_hyperparameters(hparams)

    model_save_path = os.path.join(logger.get_log_path(), "best_model.pth")
    best_val_metric = -float('inf')  # Track the best Recall@1_I2T

    print("Setting up Datasets")
    dataset = Coco2014Dataset(
        train_encodings_path="coco2014/label_encodings.pt")

    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()

    model = CLIPModel(
        projection_dim=hparams['projection_dim'],
        freeze_backbone=hparams['freeze_backbone']
    ).to(device)

    loss_fn = InfoNCELoss(temperature=hparams['temperature'])

    optimizer = optim.AdamW(
        model.parameters(),
        lr=hparams['learning_rate'],
        weight_decay=hparams['weight_decay']
    )

    print("Model and Optimizer initialized. Starting training...")

    # --- 4. Main Training Loop ---
    for epoch in range(1, hparams['epochs'] + 1):
        start_time = time.time()

        # ==============
        # TRAINING STEP
        # ==============
        model.train()
        total_train_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {
                          epoch+1}/{hparams['epochs']} [TRAIN]", leave=False)
        for images, embeddings in train_loop:
            images, embeddings = images.to(device), embeddings.to(device)

            optimizer.zero_grad()
            image_features = model(images)

            loss = loss_fn(image_features, embeddings)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # ==============
        # VALIDATION STEP
        # ==============
        model.eval()

        all_val_image_features = []
        all_val_text_features = []

        # Placeholder for actual validation loop (replace with data loader iteration)
        total_val_loss = 0.0
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {
                epoch+1}/{hparams['epochs']} [VAL]", leave=False)
            for images, embeddings in val_loop:
                images, embeddings = images.to(device), embeddings.to(device)

                image_features = model(images)
                val_loss = loss_fn(image_features, embeddings)
                total_val_loss += val_loss.item()

                all_val_image_features.append(image_features.cpu())
                all_val_text_features.append(embeddings.cpu())

        avg_val_loss = total_val_loss / len(val_loader.dataset)

        # Compute Recall@K on accumulated features
        val_image_features_all = torch.cat(all_val_image_features, dim=0)
        val_text_features_all = torch.cat(all_val_text_features, dim=0)

        recall_metrics = calculate_recall_at_k(
            val_image_features_all, val_text_features_all)

        # --- 5. Logging and Checkpointing ---

        # Combine all metrics for logging
        metrics = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            **recall_metrics
        }

        logger.save_metrics(epoch, metrics)

        # Checkpointing (using Recall@1_I2T as the primary metric)
        current_metric = recall_metrics.get(
            'R@1_I2T', 0.0)  # Fallback to 0 if not found
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            torch.save(model.state_dict(), model_save_path)
            print(f"SAVED checkpoint. New best R@1_I2T: {best_val_metric:.4f}")

        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{hparams['epochs']} | Time: {epoch_time:.2f}s | Train Loss: {
              avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | R@1_I2T: {current_metric:.4f}")

    # --- 6. Finalization ---
    print("Training finished.")
    logger.plot_loss_curves(show_plot=False, save_plot=True)
    print(f"Loss curves saved to {logger.get_log_path()}")


if __name__ == '__main__':
    hparams = {
        'epochs': 80,
        'learning_rate': 0.003,
        'weight_decay':  0.0003,
        'batch_size': 128,
        'exp_name': "ShitCrap",
        'projection_dim': 512,
        'temperature': 1,
        'freeze_backbone': True,
        'num_workers': 8
    }

    train_and_validate(hparams)
