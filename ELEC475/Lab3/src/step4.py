import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from train_model import evaluate_model, load_teacher_model, load_student_model, load_pascal_voc_data, plot_history
from torch.utils.data import DataLoader
from model import StudentModel
import matplotlib.pyplot as plt
from fcnresnet import FCNResNetWrapper

# Set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The PASCAL VOC 2012 dataset has 21 classes
NUM_CLASSES = 21

# --- KD Hyperparameters (Based on standard practice and Lab requirements) ---
ALPHA = 1.0  # Weight for standard CE loss (L_CE)
BETA = 1.0   # Weight for response-based KD loss (L_KD)
GAMMA = 0.5  # Weight for feature-based KD loss (L_Feature)
TAU = 4.0    # Temperature (tau) for softening logits

# --- Hyperparameters from Best Supervised Run ---
LEARNING_RATE = 0.01
NUM_KD_EPOCHS = 4

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
# 2. KD HELPER FUNCTION
# ====================================================================


def softmax_with_temp(logits, temperature):
    """Applies Hinton-style softmax with a specified temperature (tau)."""
    return F.softmax(logits / temperature, dim=1)

# ====================================================================
# 3. KD TRAINING FUNCTION (Modified to return history)
# ====================================================================


def train_kd_model(teacher_model, student_model, train_loader, val_loader,
                   mode='response', num_epochs=20, lr=0.01, model_name="Student_KD"):

    print(f"\n--- 6. Starting Knowledge Distillation Training ({mode}) ---")

    # Setup
    teacher_model.eval()
    student_model.train()
    optimizer = optim.SGD(student_model.parameters(),
                          lr=lr, momentum=0.9, weight_decay=1e-4)
    best_miou = 0.0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_miou': []
    }
    # --- Feature Projection Setup (NEW) ---
    projection_modules = nn.ModuleDict()
    projection_optimizer = None  # Define optimizer for the projection layers later

    if mode == 'feature' or mode == 'both':
        # Temporarily run a dummy batch through to determine feature shapes
        # (If you know the shapes, you can hardcode them, but this is safer)
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 256, 256).to(
                DEVICE)  # Assuming 256x256 input size
            student_dummy_out = student_model(dummy_input)
            teacher_dummy_out = teacher_model(
                dummy_input)  # Requires FCNResNetWrapper!

        # Determine required projections
        for tap_name in ['low', 'mid', 'high']:
            Fs_channels = student_dummy_out['features'][tap_name].shape[1]
            Ft_channels = teacher_dummy_out['features'][tap_name].shape[1]

            if Fs_channels != Ft_channels:
                print(f"Adding Projection for {tap_name}: {
                      Fs_channels} -> {Ft_channels}")
                # Create a 1x1 convolution to match the channel count
                proj = nn.Conv2d(Fs_channels, Ft_channels,
                                 kernel_size=1).to(DEVICE)
                projection_modules[tap_name] = proj

        # Combine the parameters of the Student Model and the new Projection Modules
        all_params = list(student_model.parameters()) + \
            list(projection_modules.parameters())
        # <-- UPDATE optimizer definition!
        optimizer = optim.SGD(
            all_params, lr=lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(num_epochs):
        total_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {
                          epoch+1}/{num_epochs} [KD {mode}]", leave=False)

        for images, targets in train_loop:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()

            # --- Forward Pass and Loss Calculation (as detailed previously) ---
            student_output = student_model(images)
            with torch.no_grad():
                teacher_output = teacher_model(images)

            logits_s = student_output['out']
            loss_ce = CE_Loss(logits_s, targets)

            # Response-Based KD (L_KD)
            loss_kd = torch.tensor(0.0).to(DEVICE)
            if mode == 'response' or mode == 'both':
                logits_t = teacher_output['out']
                soft_targets = softmax_with_temp(logits_t, TAU)
                soft_prob_s = softmax_with_temp(logits_s, TAU)
                loss_kd = F.kl_div(soft_prob_s.log(),
                                   soft_targets, reduction='batchmean')

            # --- 3. Feature-Based KD Loss (L_Feature) ---
            loss_feature = torch.tensor(0.0).to(DEVICE)
            if mode == 'feature' or mode == 'both':
                target_one = torch.ones(images.size(0)).to(DEVICE)

                for tap_name in ['low', 'mid', 'high']:
                    Fs = student_output['features'][tap_name]
                    Ft = teacher_output['features'][tap_name]

                    # --- NEW: APPLY FEATURE PROJECTION ---
                    if tap_name in projection_modules:
                        Fs = projection_modules[tap_name](Fs)

                    # --- Existing spatial interpolation (still needed) ---
                    if Fs.shape[2:] != Ft.shape[2:]:
                        Fs = F.interpolate(
                            Fs, size=Ft.shape[2:], mode='bilinear', align_corners=False)

                    # Flatten and calculate loss (shapes should now match)
                    Fs_flat = Fs.flatten(start_dim=1)
                    Ft_flat = Ft.flatten(start_dim=1)
                    loss_feature += Feature_Loss(Fs_flat, Ft_flat, target_one)

                loss_feature = loss_feature / 3.0

            # Joint Loss (L)
            loss = ALPHA * loss_ce
            if mode == 'response' or mode == 'both':
                loss += BETA * loss_kd
            if mode == 'feature' or mode == 'both':
                loss += GAMMA * loss_feature

            # Backpropagation
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            train_loop.set_postfix(loss=loss.item())

        # Calculate and log training loss for the epoch
        avg_train_loss = total_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)

        # --- Validation after each epoch ---
        # evaluate_model is assumed to return (miou, speed, loss)
        current_miou, _, avg_val_loss = evaluate_model(
            student_model,
            val_loader,
            model_name=f"KD {mode} (Validation)",
            is_validation=True
        )
        history['val_miou'].append(current_miou)
        history['val_loss'].append(avg_val_loss)

        # Save checkpoint
        if current_miou > best_miou:
            best_miou = current_miou
            torch.save(student_model.state_dict(), f'{model_name}_best.pth')

    return f'{model_name}_best.pth', history


def main():
    # --- SETUP (Assume these utility functions are defined elsewhere) ---
    # Load the FCN-ResNet50 Teacher model
    # teacher_model = load_teacher_model()
    teacher_model = FCNResNetWrapper(NUM_CLASSES).to(DEVICE)
    teacher_model.eval()

    # Create the Student Model architecture
    student_model_arch = load_student_model()

    # Load DataLoaders
    train_loader, val_loader = load_pascal_voc_data(batch_size=8)

    # --- Supervised Checkpoint Name ---
    SUPERVISED_CHECKPOINT = 'trained/Student_Supervised_best.pth'

    # --- EXPERIMENT 1: Response-Based KD ---
    print("\n--- EXPERIMENT 1: Response-Based Knowledge Distillation ---")
    student_model_resp = student_model_arch  # Start with fresh architecture
    student_model_resp.load_state_dict(torch.load(SUPERVISED_CHECKPOINT))

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
    print("\n--- EXPERIMENT 2: Feature-Based Knowledge Distillation ---")
    student_model_feat = student_model_arch  # Start with fresh architecture
    student_model_feat.load_state_dict(torch.load(SUPERVISED_CHECKPOINT))

    kd_feat_path, kd_feat_history = train_kd_model(
        teacher_model,
        student_model_feat,
        train_loader,
        val_loader,
        mode='feature',
        num_epochs=NUM_KD_EPOCHS,
        lr=LEARNING_RATE,
        model_name="Student_KD_Feature"
    )
    plot_history(kd_feat_history, "KD Feature-Based")

    # --- EVALUATION AND REPORT GENERATION (User's responsibility to fill in) ---
    print("\n\n--- Step Four Complete. Gather results for the Report Table ---")
    # You would typically call evaluate_model() on the best paths here to get final metrics.
    # E.g., final_miou, speed, _ = evaluate_model(model_resp, val_loader, ...)


if __name__ == '__main__':
    main()
