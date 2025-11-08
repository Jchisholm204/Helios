import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchmetrics.segmentation import MeanIoU

# Set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The PASCAL VOC 2012 dataset has 21 classes
NUM_CLASSES = 21


def calculate_miou(predictions, targets, num_classes, device="cuda"):
    preds = predictions.argmax(dim=1)

    # Compute mean IoU
    metric = MeanIoU(num_classes=num_classes).to(device)
    mean_iou = metric(preds, targets)

    # Compute per-class intersection and union manually
    intersection = torch.zeros(num_classes, dtype=torch.long, device=device)
    union = torch.zeros(num_classes, dtype=torch.long, device=device)

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        tp = (pred_cls & target_cls).sum()
        union_cls = (pred_cls | target_cls).sum()

        intersection[cls] = tp
        union[cls] = union_cls

    return intersection.cpu().numpy(), union.cpu().numpy(), mean_iou.item()


# def calculate_miou(predictions, targets, num_classes):
#     """
#     Calculates the Mean Intersection over Union (mIoU) for a batch of predictions.
#
#     (Same implementation as before - computes intersection and union for each class)
#     """
#     preds = predictions.argmax(dim=1)
#     mask = (targets >= 0) & (targets < num_classes)
#
#     preds = preds[mask].flatten()
#     targets = targets[mask].flatten()
#
#     intersection = torch.zeros(num_classes, dtype=torch.long).to(DEVICE)
#     union = torch.zeros(num_classes, dtype=torch.long).to(DEVICE)
#
#     if preds.numel() > 0:
#         for cls in range(num_classes):
#             pred_cls = (preds == cls)
#             target_cls = (targets == cls)
#
#             tp = (pred_cls & target_cls).sum().item()
#             intersection[cls] = tp
#             union[cls] = (pred_cls | target_cls).sum().item()
#
#     return intersection.cpu().numpy(), union.cpu().numpy()


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
    print(f"\n--- 3. Starting Evaluation for {model_name} ---")

    total_intersection = np.zeros(NUM_CLASSES)
    total_union = np.zeros(NUM_CLASSES)

    # Initialize MeanIoU (no ignore_index in older versions)
    miou_metric = MeanIoU(num_classes=NUM_CLASSES).to(DEVICE)

    start_time = torch.cuda.Event(enable_timing=True) if DEVICE.type == 'cuda' else None
    end_time = torch.cuda.Event(enable_timing=True) if DEVICE.type == 'cuda' else None
    total_time_ms = 0
    total_images = 0

    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc=f"Evaluating {model_name}"):
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            # Optional: skip ignored labels (e.g., 255)
            ignore_mask = targets != 255
            targets = torch.where(ignore_mask, targets, torch.zeros_like(targets))

            if DEVICE.type == 'cuda':
                start_time.record()

            output = model(images)['out']

            if DEVICE.type == 'cuda':
                end_time.record()
                torch.cuda.synchronize()
                total_time_ms += start_time.elapsed_time(end_time)

            preds = output.argmax(dim=1)

            # Apply same ignore mask to predictions
            preds = preds * ignore_mask

            # Update metric
            miou_metric.update(preds, targets)

            # Per-class intersection/union
            for cls in range(NUM_CLASSES):
                pred_cls = (preds == cls)
                target_cls = (targets == cls)
                tp = (pred_cls & target_cls).sum().item()
                union_cls = (pred_cls | target_cls).sum().item()
                total_intersection[cls] += tp
                total_union[cls] += union_cls

            total_images += images.size(0)

    torchmetrics_miou = miou_metric.compute().item()
    epsilon = 1e-6
    class_iou = total_intersection / (total_union + epsilon)
    valid_classes = total_union > 0
    manual_miou = np.mean(class_iou[valid_classes])

    print(f"\n--- Results: {model_name} ---")
    print(f"TorchMetrics mIoU: **{torchmetrics_miou:.4f}**")
    print(f"Manual mIoU (for comparison): **{manual_miou:.4f}**")

    if total_images > 0 and total_time_ms > 0:
        avg_time_ms_per_image = total_time_ms / total_images
        print(f"Inference Speed: **{avg_time_ms_per_image:.2f} msec per image** (on {DEVICE})")
    else:
        avg_time_ms_per_image = None
        print("Inference Speed: Not timed (or 0 images processed).")

    print("---------------------------------------")
    return torchmetrics_miou, avg_time_ms_per_image


def main():
    # 1. Load the Model
    teacher_model = load_teacher_model()

    # 2. Load the Data
    val_loader = load_pascal_voc_data(batch_size=8)

    # 3. Evaluate and Print Results
    miou, speed = evaluate_model(
        teacher_model, val_loader, model_name="FCN-ResNet50 Teacher")

    print("\nProgram finished successfully.")


if __name__ == '__main__':
    main()
