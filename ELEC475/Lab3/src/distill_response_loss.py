import torch
import torch.nn as nn
import torch.optim as optim
# Response based learning
from tqdm import tqdm
import json
import torch.nn.functional as F
from train_segmodel import poly_lr_scheduler, calculate_miou


def response_based_kd_loss(student_logits, teacher_logits, ground_truth_labels,
                           alpha, beta, temperature):
    """
    Response-based knowledge distillation loss for segmentation.
    """

    # Standard supervised CE loss
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)(
        student_logits, ground_truth_labels)

    # Softened distributions
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    log_soft_student = F.log_softmax(student_logits / temperature, dim=1)

    # KL divergence normalized by pixel count
    kd_loss = F.kl_div(
        log_soft_student, soft_teacher, reduction='batchmean'
    ) * (temperature ** 2)

    # Normalize by spatial size to avoid massive scale
    num_pixels = ground_truth_labels.numel()
    kd_loss = kd_loss / num_pixels

    total_loss = alpha * ce_loss + beta * kd_loss
    return total_loss


def distill_model(model: nn.Module, teacher_model: nn.Module,
                  train_loader: torch.utils.data.DataLoader,
                  val_loader: torch.utils.data.DataLoader,
                  alpha: float, beta: float, temp: float,
                  n_epochs: int, lr=1e-3, decay=1e-4,
                  save_path="undefined", device='cuda'):
    optimizer_fn = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    history = {
        'train_type': 'response_based_learning',
        'params': {
            'alpha': alpha,
            'beta': beta,
            'temp': temp,
            'lr': lr,
            'decay': decay
        },
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
        accum_loss = 0.0
        accum_miou = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {
                          epoch+1}/{n_epochs} [TRAIN]", leave=False)
        for images, targets in train_loop:
            images = images.to(device)
            targets = targets.to(device)

            optimizer_fn.zero_grad()
            student_outputs = model(images)['out']
            with torch.no_grad():
                teacher_outputs = teacher_model(images)['out']
            total_loss = response_based_kd_loss(
                student_outputs, teacher_outputs, targets, alpha, beta, temp)
            total_loss.backward()
            optimizer_fn.step()

            accum_loss += total_loss.item()
            accum_miou += calculate_miou(student_outputs.detach(), targets)

        avg_train_loss = accum_loss / len(train_loader)
        avg_train_miou = accum_miou / len(train_loader)
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
                val_loss_fn = nn.CrossEntropyLoss(ignore_index=255)
                loss = val_loss_fn(outputs, targets)
                val_loss += loss.item()
                val_miou += calculate_miou(outputs, targets)

        avg_val_loss = val_loss / len(val_loader)
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
            print(f"Saved Model with new best Val Loss: {best_loss:.4f}")

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
