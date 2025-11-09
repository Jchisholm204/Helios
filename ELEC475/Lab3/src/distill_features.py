import torch
import torch.nn as nn
import torch.optim as optim
# Response based learning
from tqdm import tqdm
import json
from train_segmodel import poly_lr_scheduler, calculate_miou
import torch.nn.functional as F


class FeatureKDLoss(nn.Module):
    def __init__(self, feature_keys=['low', 'mid', 'high']):
        super().__init__()
        self.feature_keys = feature_keys
        # 1x1 conv projections per feature tap
        self.projections = nn.ModuleDict()

    def forward(self, student_features: dict, teacher_features: dict):
        total_loss = 0.0
        B = None  # batch size

        for key in self.feature_keys:
            s_feat = student_features[key]
            t_feat = teacher_features[key].detach()

            if B is None:
                B = s_feat.size(0)

            # 1x1 projection to match teacher channels
            if key not in self.projections:
                self.projections[key] = nn.Conv2d(s_feat.size(
                    1), t_feat.size(1), kernel_size=1).to(s_feat.device)
            s_feat = self.projections[key](s_feat)

            # Global average pooling to get one vector per sample
            s_vec = F.adaptive_avg_pool2d(s_feat, (1, 1)).reshape(B, -1)
            t_vec = F.adaptive_avg_pool2d(t_feat, (1, 1)).reshape(B, -1)

            # Cosine similarity: 1 - similarity gives loss
            cos_sim = F.cosine_similarity(s_vec, t_vec, dim=1)
            loss = 1 - cos_sim.mean()
            total_loss += loss

        return total_loss


def distill_model(model: nn.Module, teacher_model: nn.Module,
                  train_loader: torch.utils.data.DataLoader,
                  val_loader: torch.utils.data.DataLoader,
                  alpha: float, beta: float, temp: float,
                  n_epochs: int, lr=1e-3, decay=1e-4,
                  save_path="undefined", device='cuda'):
    optimizer_fn = optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    teacher_model.eval()
    feature_fn = FeatureKDLoss()
    scaler = torch.cuda.amp.GradScaler()
    for param in teacher_model.parameters():
        param.requires_grad = False

    history = {
        'train_type': 'feature_based_learning',
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
            with torch.cuda.amp.autocast():
                student_outputs = model(images)
                teacher_model.eval()
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)
                # Standard loss
                standard_loss = loss_fn(student_outputs['out'], targets)
                # Feature based losses
                feature_loss = feature_fn.forward(
                    student_outputs, teacher_outputs['features'])
                batch_loss_tensor = (alpha * standard_loss) + \
                    (beta * feature_loss)
                batch_loss_tensor = batch_loss_tensor * temp

            scaler.scale(batch_loss_tensor).backward()
            scaler.step(optimizer_fn)
            scaler.update()

            accum_loss += batch_loss_tensor.item()
            accum_miou += calculate_miou(
                student_outputs['out'].detach(), targets)

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
