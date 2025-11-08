import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from model import StudentModel

# Set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The PASCAL VOC 2012 dataset has 21 classes
NUM_CLASSES = 21


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


def load_pascal_voc_data(batch_size=8, image_size=(520, 520)):
    """
    Loads the PASCAL VOC 2012 validation dataset and prepares the DataLoader.

    Args:
        batch_size (int): The batch size for the DataLoader.
        image_size (tuple): The size to which images and targets are resized.

    Returns:
        torch.utils.data.DataLoader: The validation DataLoader.
    """
    print("\n--- 2. Loading PASCAL VOC 2012 Validation Dataset ---")

    # ImageNet Mean/Std for normalization required by pretrained models
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Transforms for input images
    eval_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Transforms for ground truth segmentation masks
    # Resize uses InterpolationMode.NEAREST to keep class labels discrete
    target_transforms = transforms.Compose([
        transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        # Convert to LongTensor and scale back: (0-1) -> (0-255) -> (0-20, 255)
        transforms.Lambda(lambda x: (x.squeeze(0) * 255).long())
    ])

    print("Attempting to load datset")
    val_dataset = datasets.VOCSegmentation(
        root='./data',
        year='2012',
        image_set='val',
        download=False,
        transform=eval_transforms,
        target_transform=target_transforms
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4  # Adjust based on your system
    )

    print(f"Validation dataset size: {len(val_dataset)} images.")
    print(f"DataLoader batch size: {batch_size}.")
    return val_loader


def evaluate_model(model, data_loader, model_name="Model"):
    """
    Runs the evaluation loop, calculates accumulated mIoU metrics.

    Args:
        model (torch.nn.Module): The segmentation model to evaluate.
        data_loader (DataLoader): The validation/test data loader.
        model_name (str): Name for printouts.

    Returns:
        float: The final calculated mIoU.
        float: The inference speed in msec per image.
    """
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

    return final_miou, avg_time_ms_per_image if 'avg_time_ms_per_image' in locals() else None


def main():
    # 1. Load the Model
    teacher_model = load_teacher_model()

    student_model = load_student_model()

    # 2. Load the Data
    val_loader = load_pascal_voc_data(batch_size=8)

    # 3. Evaluate and Print Results
    miou, speed = evaluate_model(
        teacher_model, val_loader, model_name="FCN-ResNet50 Teacher")

    miou, speed = evaluate_model(
        student_model, val_loader, model_name="Student (untrained)")

    print("\nProgram finished successfully.")


if __name__ == '__main__':
    main()
