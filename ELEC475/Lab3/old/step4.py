import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
# Import utilities and models
# Assuming load_teacher_model is removed/integrated
from train_model import evaluate_model, load_student_model, load_pascal_voc_data, plot_history
# Assuming this is your base student architecture function/class
from model import StudentModel
from fcnresnet import FCNResNetWrapper
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import os

# Set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The PASCAL VOC 2012 dataset has 21 classes
NUM_CLASSES = 21

# ====================================================================
# KD Hyperparameters (Consolidated from kd_config.py)
# ====================================================================

# Learning Rate: Restore the effective high LR that gave the 0.43 baseline.
LEARNING_RATE = 0.0005

# Number of Epochs for KD Training (Must be increased from 4)
NUM_KD_EPOCHS = 20

# Temperature for Softmax (Standard Hinton value)
TAU = 4.0

# Weight for Standard Cross-Entropy Loss (L_CE)
ALPHA = 1.0

# Weight for Response-Based KD Loss (L_KD - KL Divergence)
# Reduced from 1.0 to 0.1 for stability.
BETA = 0.0005

# Weight for Feature-Based KD Loss (L_Feature - Cosine Embedding Loss)
# Reduced drastically from 0.5 (which caused collapse) to 0.005 for stability.
GAMMA = 0.005

# Loss functions (use CrossEntropyLoss for standard and KD, CosineEmbeddingLoss for features)
CE_Loss = nn.CrossEntropyLoss(ignore_index=255)
Feature_Loss = nn.CosineEmbeddingLoss(reduction='mean')
# Assume global constants: DEVICE, NUM_CLASSES, evaluate_model, tqdm, torch.save


def softmax_with_temp(logits, temperature):
    """
    Applies Hinton-style softmax with a specified temperature (tau).
    Used to soften the Teacher's and Student's logit distributions.
    """
    # Softmax formula: exp(z_i / T) / sum(exp(z_j / T))
    return F.softmax(logits / temperature, dim=1)


def plot_history(history, model_name="Model"):
    """
    Generates and saves two separate plots: Loss and mIoU History.
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

    # --- Plot 2: mIoU History ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_miou'], 'r-', label='Validation mIoU')
    # Note: Training mIoU is NOT collected in KD runs for simplicity
    plt.title(f'{model_name} Validation mIoU History')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_filename = f'{model_name}_kd_metrics.png'
    plt.savefig(plot_filename)
    print(f"\n✅ Training plot saved to: {plot_filename}")
    plt.close()

# ====================================================================
# 3. KD TRAINING FUNCTION (Copied from step4_kd_training.py for completeness)
# ====================================================================

# Note: The following function should be in step4_kd_training.py,
# but included here for context of the main function changes.


def train_kd_model(teacher_model, student_model, train_loader, val_loader,
                   mode='response', num_epochs=40, lr=0.01, model_name="Student_KD"):

    # --- 1. MANDATORY CHECKPOINT LOAD (Moved to main for clarity, but kept here as safeguard if the main load fails) ---
    # NOTE: In 'main', this is set to 'trained/Student_Supervised_best.pth'
    SUPERVISED_CHECKPOINT_PATH = 'Student_best.pth'

    # We rely on the main function to load the checkpoint now.

    print(f"\n--- 6. Starting Knowledge Distillation Training ({mode}) ---")

    # Setup
    teacher_model.eval()
    student_model.train()

    # --- Feature Projection Setup (MUST BE HERE before optimizer) ---
    projection_modules = nn.ModuleDict()

    if mode == 'feature' or mode == 'both':
        # Use a dummy forward pass to get shapes:
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 256, 256).to(DEVICE)
            student_dummy_out = student_model(dummy_input)
            teacher_dummy_out = teacher_model(dummy_input)

        for tap_name in ['low', 'mid', 'high']:
            Fs_channels = student_dummy_out['features'][tap_name].shape[1]
            Ft_channels = teacher_dummy_out['features'][tap_name].shape[1]

            if Fs_channels != Ft_channels:
                print(f"Adding Projection for {tap_name}: {
                      Fs_channels} -> {Ft_channels}")
                proj = nn.Conv2d(Fs_channels, Ft_channels,
                                 kernel_size=1).to(DEVICE)
                projection_modules[tap_name] = proj

        all_params = list(student_model.parameters()) + \
            list(projection_modules.parameters())
        optimizer = optim.SGD(
            all_params, lr=lr, momentum=0.9, weight_decay=1e-3)
    else:
        optimizer = optim.SGD(student_model.parameters(),
                              lr=lr, momentum=0.9, weight_decay=1e-3)
        # Add scheduler RIGHT HERE
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=lr * 0.01  # End at 1% of starting LR
        )

    # --- AMP Setup: Initialize the scaler for mixed precision training ---
    scaler = GradScaler()
    best_miou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_miou': []}

    for epoch in range(num_epochs):
        total_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {
                          epoch+1}/{num_epochs} [KD {mode}]", leave=False)

        for images, targets in train_loop:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                # --- Forward Pass and Loss Calculation (as detailed previously) ---
                student_output = student_model(images)
                with torch.no_grad():
                    teacher_output = teacher_model(images)

                logits_s = student_output['out']
                loss_ce = CE_Loss(logits_s, targets)

                # Response-Based KD (L_KD)
                loss_kd = torch.tensor(0.0).to(DEVICE)
                # if mode == 'response' or mode == 'both':
                #     logits_t = teacher_output['out']
                #     soft_targets = softmax_with_temp(logits_t, TAU)
                #     soft_prob_s = softmax_with_temp(logits_s, TAU)
                #     loss_kd = F.kl_div(soft_prob_s.log(),
                #                        soft_targets, reduction='batchmean')
                if mode == 'response' or mode == 'both':
                    logits_t = teacher_output['out']
                    B, C, H, W = logits_s.shape
                    logits_s_flat = logits_s.permute(0, 2, 3, 1).reshape(-1, C)
                    logits_t_flat = logits_t.permute(0, 2, 3, 1).reshape(-1, C)

                    # CORRECT: Use log_softmax directly, not softmax().log()
                    student_log_probs = F.log_softmax(
                        logits_s_flat / TAU, dim=1)
                    teacher_probs = F.softmax(logits_t_flat / TAU, dim=1)

                    # Apply temperature scaling to KD loss
                    loss_kd = (TAU ** 2) * F.kl_div(
                        student_log_probs,
                        teacher_probs,
                        reduction='batchmean'
                    )

                # --- 3. Feature-Based KD Loss (L_Feature) ---
                loss_feature = torch.tensor(0.0).to(DEVICE)
                if mode == 'feature' or mode == 'both':
                    target_one = torch.ones(images.size(0)).to(
                        DEVICE)

                    for tap_name in ['low', 'mid', 'high']:
                        Fs = student_output['features'][tap_name]
                        Ft = teacher_output['features'][tap_name]

                        # --- APPLY FEATURE PROJECTION ---
                        if tap_name in projection_modules:
                            Fs = projection_modules[tap_name](Fs)

                        # --- Spatial interpolation (still needed) ---
                        if Fs.shape[2:] != Ft.shape[2:]:
                            Fs = F.interpolate(
                                Fs, size=Ft.shape[2:], mode='bilinear', align_corners=False)

                        # Flatten and calculate loss (shapes should now match)
                        Fs_flat = Fs.flatten(start_dim=1)
                        Ft_flat = Ft.flatten(start_dim=1)
                        loss_feature += Feature_Loss(Fs_flat,
                                                     Ft_flat, target_one)

                    loss_feature = loss_feature / 3.0  # Average over 3 taps

                # Joint Loss (L)
                loss = ALPHA * loss_ce
                if mode == 'response' or mode == 'both':
                    loss += BETA * loss_kd
                if mode == 'feature' or mode == 'both':
                    loss += GAMMA * loss_feature

            # Backpropagation with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            train_loop.set_postfix(loss=loss.item())

        # Calculate and log training loss for the epoch
        avg_train_loss = total_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)

        # --- Validation after each epoch ---
        current_miou, _, avg_val_loss = evaluate_model(
            student_model,
            val_loader,
            model_name=f"KD {mode} (Validation)",
            is_validation=True
        )
        history['val_miou'].append(current_miou)
        history['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # Print metrics (essential for tracking progress)
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {
              avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val mIoU: {current_miou:.4f}")

        # Save checkpoint
        if current_miou > best_miou:
            best_miou = current_miou
            torch.save({'model_state_dict': student_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_miou': best_miou,
                        'epoch': epoch},
                       f'{model_name}_best.pth')

    return f'{model_name}_best.pth', history


def main():
    # --- SETUP (Assume these utility functions are defined elsewhere) ---
    # Load the FCN-ResNet50 Teacher model
    teacher_model = FCNResNetWrapper(NUM_CLASSES).to(DEVICE)
    teacher_model.eval()

    # Load DataLoaders
    # NOTE: Set batch_size=8 to prevent OOM errors with the Teacher model
    train_loader, val_loader = load_pascal_voc_data(batch_size=8)

    # --- Supervised Checkpoint Name (Ensure this path is correct!) ---
    SUPERVISED_CHECKPOINT = 'pre_optim/Student_Supervised_best.pth'

    # Ensure the checkpoint exists before proceeding
    if not os.path.exists(SUPERVISED_CHECKPOINT):
        print(f"ERROR: Supervised checkpoint not found at {
              SUPERVISED_CHECKPOINT}. Cannot start KD.")
        return

    # --- EXPERIMENT 1: Response-Based KD ---
    print("\n--- EXPERIMENT 1: Response-Based Knowledge Distillation ---")

    # FIX: Instantiate a FRESH model instance for this experiment
    student_model_resp = load_student_model().to(DEVICE)
    # Load the best supervised weights
    # student_model_resp.load_state_dict(torch.load(
    #     SUPERVISED_CHECKPOINT)['model_state_dict'])
    # --- Robust Checkpoint Loading Logic ---
    checkpoint_resp = torch.load(SUPERVISED_CHECKPOINT, map_location=DEVICE)
    if isinstance(checkpoint_resp, dict) and 'model_state_dict' in checkpoint_resp:
        state_dict_to_load_resp = checkpoint_resp['model_state_dict']
    else:
        # Assume the checkpoint file contains only the raw state_dict
        state_dict_to_load_resp = checkpoint_resp

    student_model_resp.load_state_dict(state_dict_to_load_resp)
    # --- End Robust Loading ---

    kd_resp_path, kd_resp_history = train_kd_model(
        teacher_model,
        student_model_resp,
        train_loader,
        val_loader,
        mode='response',
        num_epochs=NUM_KD_EPOCHS,
        lr=LEARNING_RATE,
        model_name="Student_KD_Response"
    )
    plot_history(kd_resp_history, "KD Response-Based")

    # --- EXPERIMENT 2: Feature-Based KD ---
    # print("\n--- EXPERIMENT 2: Feature-Based Knowledge Distillation ---")
    #
    # # FIX: Instantiate a FRESH model instance for this experiment
    # student_model_feat = load_student_model().to(DEVICE)
    # # Load the best supervised weights
    # # student_model_feat.load_state_dict(torch.load(
    # #     SUPERVISED_CHECKPOINT)['model_state_dict'])
    # # Load the best supervised weights
    # # The original line: student_model_feat.load_state_dict(torch.load(
    # # SUPERVISED_CHECKPOINT)['model_state_dict']) caused KeyError.
    # # We use robust loading logic instead:
    # # --- Robust Checkpoint Loading Logic ---
    # checkpoint_feat = torch.load(SUPERVISED_CHECKPOINT, map_location=DEVICE)
    # if isinstance(checkpoint_feat, dict) and 'model_state_dict' in checkpoint_feat:
    #     # Case 1: Checkpoint is a dictionary containing 'model_state_dict'
    #     state_dict_to_load_feat = checkpoint_feat['model_state_dict']
    # else:
    #     # Case 2: Checkpoint file contains only the raw state_dict
    #     state_dict_to_load_feat = checkpoint_feat
    #
    # student_model_feat.load_state_dict(state_dict_to_load_feat)
    # # --- End Robust Loading ---
    #
    # kd_feat_path, kd_feat_history = train_kd_model(
    #     teacher_model,
    #     student_model_feat,
    #     train_loader,
    #     val_loader,
    #     mode='feature',
    #     num_epochs=NUM_KD_EPOCHS,
    #     lr=LEARNING_RATE,
    #     model_name="Student_KD_Feature"
    # )
    # plot_history(kd_feat_history, "KD Feature-Based")

    # --- EVALUATION AND REPORT GENERATION (User's responsibility to fill in) ---
    print("\n\n--- Step Four Complete. Gather results for the Report Table ---")


if __name__ == '__main__':
    main()

