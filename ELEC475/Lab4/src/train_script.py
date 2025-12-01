import argparse
import torch
import torch.optim as optim
import time
import os
from typing import Dict, Any
from tqdm import tqdm
import random
from torchvision import transforms

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

    img_transform = None
    # img_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(
    #         size=224,
    #         scale=(0.8, 1.0),
    #         ratio=(0.9, 1.1)
    #     ),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ColorJitter(
    #         brightness=0.1,
    #         contrast=0.1,
    #         saturation=0.1,
    #         hue=0.02
    #     ),
    #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
    # ])

    print("Setting up Datasets")
    dataset = Coco2014Dataset(
        train_encodings_path="coco2014/train_caption_encodings.pt",
        train_transform=img_transform,
        val_encodings_path="coco2014/val_caption_encodings.pt",
        batch_size=hparams['batch_size'],
        n_workers=hparams['num_workers'],
        pin=True
    )

    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()

    model: CLIPModel = CLIPModel(
        projection_dim=hparams['projection_dim'],
        freeze_backbone=hparams['freeze_backbone']
    ).to(device)

    # Correct initialization using the hyperparameter
    loss_fn = InfoNCELoss(temperature=hparams['temperature'])

    param_groups = [
        {'params': model.image_encoder.parameters(),
         'lr': hparams['backbone_lr']},
        {'params': model.projection_head.parameters(),
         'lr': hparams['head_lr']},
    ]

    optimizer = optim.AdamW(
        param_groups,
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
            epoch}/{hparams['epochs']} [TRAIN]", leave=False)

        # Load embeddings
        train_dataset = dataset.train_ds
        train_dataset._init_self()
        train_encodings = train_dataset.encodings
        for images, img_ids in train_loop:
            captions = []
            for img_id in img_ids:
                captions.append(random.choice(train_encodings[int(img_id)]))
            embeddings = torch.stack(captions).to(device)
            images = images.to(device)

            # images, embeddings = images.to(device), embeddings.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                image_features = model(images)

                loss = loss_fn(image_features, embeddings)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ==============
        # VALIDATION STEP
        # ==============
        model.eval()

        val_dataset = dataset.val_ds
        val_dataset._init_self()
        val_encodings = val_dataset.encodings

        val_img_ids = list(val_dataset.image_ids)

        text_features_list = []
        for iid in val_img_ids:
            variants = val_encodings[iid]
            stacked = torch.stack(variants)
            mean_emb = stacked.mean(dim=0)
            text_features_list.append(mean_emb)

        text_features_all = torch.stack(text_features_list).to(device)
        val_imgid_index = {int(img_id): idx for idx,
                           img_id in enumerate(val_img_ids)}

        total_val_loss = 0.0
        hits = {1: 0, 5: 0, 10: 0}
        N = len(val_loader.dataset)

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {
                epoch}/{hparams['epochs']} [VAL]", leave=False)
            for images, img_ids in val_loop:
                images = images.to(device)
                captions = []
                indices = []
                for img_id in img_ids:
                    indices.append(val_imgid_index[int(img_id)])
                    captions.append(
                        text_features_all[val_imgid_index[int(img_id)]].cpu())

                embeddings = torch.stack(captions).to(device)

                # Forwards Pass
                with torch.cuda.amp.autocast():
                    image_features = model(images)
                    val_loss = loss_fn(image_features, embeddings)
                total_val_loss += val_loss.item()

                img_norm = torch.nn.functional.normalize(image_features, dim=1)
                text_norm = torch.nn.functional.normalize(
                    text_features_all, dim=1)

                # Nearest K
                sim = img_norm @ text_norm.T
                true_idx_tensor = torch.tensor(
                    indices, device=device, dtype=torch.long)   # [B]

                for k in hits.keys():
                    topk = sim.topk(k, dim=1).indices
                    hits[k] += (topk == true_idx_tensor.unsqueeze(1)
                                ).any(dim=1).sum().item()

        # FIX: Use len(val_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        recall_metrics = {f'I2T_R@{k}': hits[k]/N for k in hits.keys()}

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
            'I2T_R@1', 0.0)  # Fallback to 0 if not found
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
    logger.plot_loss_curves(show_plot=True, save_plot=True)
    print(f"Loss curves saved to {logger.get_log_path()}")
    model_save_path = os.path.join(logger.get_log_path(), "final_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Final Model saved to {model_save_path}")


if __name__ == '__main__':
    hparams = {
        'epochs': 80,
        'backbone_lr': 0.0007,
        'head_lr': 0.008,
        'weight_decay': 0.0003,
        'batch_size': 150,
        'exp_name': "basemodel",
        'projection_dim': 512,
        'temperature': 0.04,
        'freeze_backbone': False,
        'num_workers': 16
    }

    train_and_validate(hparams)
