import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Define the number of output classes
NUM_CLASSES = 21
# Set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StudentModel(nn.Module):
    """
    Lightweight semantic segmentation model (Student) using MobileNetV3-Small backbone.
    Designed with explicit feature taps for Knowledge Distillation.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # 1. Load MobileNetV3-Small backbone (Encoder)
        # Use MobileNet_V3_Small_Weights.IMAGENET1K_V1 for pre-trained weights
        backbone = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT).features
        self.backbone = backbone

        # Define indices for feature taps (based on MobileNetV3 structure)
        # Low-level features: Typically after the first few layers (e.g., Block 2, index 2)
        # Mid-level features: Deeper layer, but before the final stages (e.g., Block 8, index 8)
        # High-level features: The final feature map before the classification head (index 12)
        self.feature_taps = {
            'low': 2,    # Channels: 16 (after index 2)
            'mid': 8,    # Channels: 48 (after index 8)
            # Channels: 576 (after index 12, the end of the backbone)
            'high': 12
        }

        # 2. Simplified Segmentation Head (Decoder)
        # Input channels for the decoder is the high-level feature channel count (576)
        # Should be 576 for small
        last_channels = self.backbone[-1].out_channels

        # We will use the final high-level feature map and pass it through a simple
        # convolutional block for the final prediction.
        self.classifier = nn.Sequential(
            # Reduce channels from 576 to a smaller intermediate size (e.g., 256)
            nn.Conv2d(last_channels, 256, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Final 1x1 convolution to get class scores (logits)
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Store intermediate feature maps (taps)
        feature_maps = {}

        # Run through the backbone layers, collecting taps
        for i, layer in enumerate(self.backbone):
            x = layer(x)

            # Check if the current layer index is one of our desired feature taps
            if i in self.feature_taps.values():
                # Find the key (low/mid/high) corresponding to the index 'i'
                tap_name = [k for k, v in self.feature_taps.items()
                            if v == i][0]
                feature_maps[tap_name] = x

        # Run through the simplified segmentation head
        # The input to the classifier is the output of the last backbone layer (high-level features)
        x = self.classifier(x)

        # Upsample the final output logits to the expected size (520x520)
        # We need the output to match the target size from the DataLoader (520, 520)
        output = F.interpolate(
            x,
            size=(520, 520),
            mode='bilinear',
            align_corners=False
        )

        # Return a dictionary matching the FCN-ResNet50 output format, plus features
        return {'out': output, 'features': feature_maps}
