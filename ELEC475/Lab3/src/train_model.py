import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from model import StudentModel
import matplotlib.pyplot as plt

# Set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The PASCAL VOC 2012 dataset has 21 classes
NUM_CLASSES = 21

# Hyperparameters (Adjust these based on your system/lab requirements)
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30


def calculate_miou(predictions, targets, num_classes):
    """
    Calculates the Mean Intersection over Union (mIoU) for a batch of predictions.

    (Same implementation as before - computes intersection and union for each class)
    """
    preds = predictions.argmax(dim=1)
    mask = (targets >= 0) & (targets < num_classes)

    preds = preds[mask].flatten()
    targets = targets[mask].flatten()

    intersection = torch.zeros(num_classes, dtype=torch.long).to(DEVICE)
    union = torch.zeros(num_classes, dtype=torch.long).to(DEVICE)

    if preds.numel() > 0:
        for cls in range(num_classes):
            pred_cls = (preds == cls)
            target_cls = (targets == cls)

            tp = (pred_cls & target_cls).sum().item()
            intersection[cls] = tp
            union[cls] = (pred_cls | target_cls).sum().item()

    return intersection.cpu().numpy(), union.cpu().numpy()


def load_teacher_model():
    """
    Loads PyTorch's pretrained FCN-ResNet50 model (the Teacher).

    Returns:
        torch.nn.Module: The loaded model set to evaluation mode.
    """
    print("--- 1. Loading pretrained FCN-ResNet50 (Teacher) ---")
    # Load the model with default pretrained weights
    teacher_model = models.segmentation.fcn_resnet50(
        weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT
    )
    teacher_model.to(DEVICE)
    # Set to evaluation mode (crucial for inference)
    teacher_model.eval()
    print(f"Teacher Model loaded on {DEVICE}.")
    print(f"Total Parameters: {sum(p.numel()
          for p in teacher_model.parameters()):,}")
    return teacher_model


def load_student_model(checkpoint_path=None):
    """
    Initializes and optionally loads weights for the custom Student model.

    Args:
        checkpoint_path (str, optional): Path to a trained checkpoint file.

    Returns:
        StudentModel: The loaded model.
    """
    print("\n--- 4. Initializing Student Model (MobileNetV3-Small based) ---")
    # Make sure your StudentModel class definition is in the script
    student_model = StudentModel(num_classes=NUM_CLASSES)

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        # Note: You will use this to load your trained model in Step 4
        student_model.load_state_dict(torch.load(checkpoint_path))

    student_model.to(DEVICE)
    # Calculate trainable parameters (p.requires_grad is important if the backbone is frozen)
    trainable_params = sum(p.numel()
                           for p in student_model.parameters() if p.requires_grad)

    print(f"Student Model initialized on {DEVICE}.")
    print(f"Total Trainable Parameters: **{trainable_params:,}**")
    return student_model


def load_pascal_voc_data(batch_size=8, image_size=(520, 520), n_workers=8):
    print("\n--- 2. Loading PASCAL VOC 2012 Dataset ---")
    # ImageNet Mean/Std for normalization required by pretrained models
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Transforms for input images (Training includes Data Augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),  # Basic Data Augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Transforms for ground truth segmentation masks (No normalization/random augmentations)
    target_transforms = transforms.Compose([
        transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x.squeeze(0) * 255).long())
    ])

    # --- Training Dataset ---
    train_dataset = datasets.VOCSegmentation(
        root='./data',
        year='2012',
        image_set='train',  # Use 'train' split
        download=False,
        transform=train_transforms,
        target_transform=target_transforms
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=n_workers,
        pin_memory=True
    )

    # --- Validation Dataset ---
    val_dataset = datasets.VOCSegmentation(
        root='./data',
        year='2012',
        image_set='val',  # Use 'val' split
        download=False,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        target_transform=target_transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True
    )

    print(f"Training dataset size: {len(train_dataset)} images.")
    print(f"Validation dataset size: {len(val_dataset)} images.")

    return train_loader, val_loader  # RETURN BOTH


criterion_eval = nn.CrossEntropyLoss(ignore_index=255)


def evaluate_model(model, data_loader, model_name="Model", is_validation=False):
    """
    Runs the evaluation loop, calculates accumulated mIoU metrics.
    Added is_validation flag to control printouts during training loop.
    """

    if not is_validation:
        print(f"\n--- 3. Starting Evaluation for {model_name} ---")

    total_intersection = np.zeros(NUM_CLASSES)
    total_union = np.zeros(NUM_CLASSES)

    # Timing variables for inference speed
    start_time = torch.cuda.Event(
        enable_timing=True) if DEVICE.type == 'cuda' else None
    end_time = torch.cuda.Event(
        enable_timing=True) if DEVICE.type == 'cuda' else None
    total_time_ms = 0
    total_images = 0
    total_loss = 0.0

    # Set model to evaluation mode and disable gradient calculation
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc=f"Evaluating {model_name}"):

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # Start timing
            if DEVICE.type == 'cuda':
                start_time.record()

            # Forward pass: FCN returns a dict, so we access ['out']
            output = model(images)['out']
            # Calculate Loss <--- New: Calculate validation/test loss
            loss = criterion_eval(output, targets)
            total_loss += loss.item() * images.size(0)

            # End timing
            if DEVICE.type == 'cuda':
                end_time.record()
                torch.cuda.synchronize()
                total_time_ms += start_time.elapsed_time(end_time)

            # Accumulate metrics
            batch_intersection, batch_union = calculate_miou(
                output, targets, NUM_CLASSES
            )
            total_intersection += batch_intersection
            total_union += batch_union
            total_images += images.size(0)

    # --- Final mIoU Calculation and Printouts ---

    epsilon = 1e-6
    class_iou = total_intersection / (total_union + epsilon)
    valid_classes = total_union > 0
    final_miou = np.mean(class_iou[valid_classes])
    avg_loss = total_loss / len(data_loader.dataset)  # <--- New: Average loss

    if total_images > 0 and total_time_ms > 0:
        # Calculate average inference speed
        avg_time_ms_per_image = total_time_ms / total_images

    if not is_validation:  # Only print full details if not validation check
        print(f"\n--- Results: {model_name} Baseline ---")
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {trainable_params:,}")
        print(f"Mean IoU (mIoU) on Validation Set: {final_miou:.4f}")

        if total_images > 0 and total_time_ms > 0:
            # Calculate average inference speed
            avg_time_ms_per_image = total_time_ms / total_images
            print(
                f"Inference Speed: {avg_time_ms_per_image:.2f} msec per image (on {DEVICE})")
        else:
            print("Inference Speed: Not timed (or 0 images processed).")
            print("---------------------------------------")
    elif is_validation:
        print(f"Validation mIoU: {final_miou:.4f}")

    return final_miou, avg_time_ms_per_image if 'avg_time_ms_per_image' in locals() else None, avg_loss


def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, model_name="Student"):
    """
    Implements the standard supervised training loop for the Student model.
    """
    print(f"\n--- 5. Starting Supervised Training for {model_name} ---")

    # Optimizer: Stochastic Gradient Descent (SGD) is common for segmentation
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    # Loss Function: Cross-Entropy Loss
    # We set ignore_index=255 to skip calculation on the PASCAL VOC 'void' label
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    best_miou = 0.0
    best_epoch = 0

    history = {
        'train_loss': [],
        'train_miou': [],
        'val_loss': [],
        'val_miou': []
    }

    # Training Loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {
                          epoch+1}/{num_epochs} [TRAIN]")
        for images, targets in train_loop:

            images = images.to(DEVICE)
            # Targets are already LongTensor (from DataLoader setup)
            targets = targets.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            # The model returns a dictionary with 'out' for the logits
            outputs = model(images)['out']

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # scheduler.step()

            total_loss += loss.item() * images.size(0)

            # Update the progress bar with the current loss
            train_loop.set_postfix(loss=loss.item())

        # --- Validation after each epoch ---
        avg_train_loss = total_loss / len(train_loader.dataset)
        print(
            f"\nEpoch {epoch+1} finished. Avg Train Loss: {avg_train_loss:.4f}")
        # history['train_loss'].append(avg_train_loss)

        # We need a separate function for validation to check mIoU
        # For simplicity, we can reuse the existing `evaluate_model` function

        # Note: You need a training DataLoader as well, not just a validation one.
        # You'll need to modify `load_pascal_voc_data` to return both loaders.

        current_miou, _, avg_val_loss = evaluate_model(
            model,
            val_loader,
            model_name="Student (Validation)",
            is_validation=True  # Add this flag to prevent verbose timing/prints during validation
        )
        history['val_miou'].append(current_miou)
        history['val_loss'].append(avg_val_loss)

        train_miou, _, train_loss = evaluate_model(
            model,
            train_loader,
            model_name="Student (Training)",
            is_validation=True  # Add this flag to prevent verbose timing/prints during validation
        )
        history['train_miou'].append(train_miou)
        history['train_loss'].append(train_loss)

        # Save the best model based on mIoU
        if current_miou > best_miou:
            best_miou = current_miou
            best_epoch = epoch
            torch.save(model.state_dict(), f'{model_name}_best.pth')
            print(f"Saved new best model with mIoU: {best_miou:.4f}")

    torch.save(model.state_dict(), f'{model_name}_final.pth')
    print(f"\nTraining complete. Best Validation mIoU: {
          best_miou:.4f} (epoch {best_epoch})")
    return f'{model_name}_best.pth', history


def plot_history(history, model_name="Model"):
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
    plot_filename = f'{model_name}_training_metrics.png'
    plt.savefig(plot_filename)
    print(f"\n✅ Training plot saved to: {plot_filename}")
    plt.show()
    plt.close()


def main():
    # 1. Load the Teacher Model (Optional for now, but good practice)
    # teacher_model = load_teacher_model()

    # 2. Load the Student Model (This checks parameter count)
    student_model = load_student_model()

    # 3. Load the Data - **Get both train and validation loaders**
    train_loader, val_loader = load_pascal_voc_data(batch_size=16)

    # 4. Train the Student Model (Step 3)
    best_checkpoint_path, history = train_model(
        student_model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        model_name="Student_Supervised"
    )

    # 5. Load the Best Trained Model and run final evaluation
    trained_student = load_student_model(checkpoint_path=best_checkpoint_path)
    final_miou, final_speed, _ = evaluate_model(
        trained_student,
        val_loader,
        model_name="Student (Final Supervised)"
    )

    plot_history(history)

    print("\nProgram finished successfully.")


if __name__ == "__main__":
    main()
