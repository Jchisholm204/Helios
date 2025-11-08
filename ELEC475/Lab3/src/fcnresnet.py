import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50
from torchvision import models

# Wrapper class to make the standard FCN-ResNet50 model return intermediate features
# as required for Feature-Based Knowledge Distillation.


class FCNResNetWrapper(nn.Module):
    """
    Wraps the torchvision FCN-ResNet50 to ensure its forward pass returns a dictionary
    containing both the final logits ('out') and the intermediate feature maps ('features').

    The feature taps correspond to the output of ResNet's layer2, layer3, and layer4.
    """

    def __init__(self, num_classes=21):
        super().__init__()
        # Load the raw pretrained FCN-ResNet50 model
        self.teacher = fcn_resnet50(
            weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT,
            num_classes=num_classes)

        # Freeze all parameters
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):

        # 1. Get the standard FCN output (final logits)
        output = self.teacher(x)

        # 2. Extract intermediate features from the ResNet50 backbone stages
        # We trace the forward path of the ResNet backbone:

        features = self.teacher.backbone.conv1(x)
        features = self.teacher.backbone.bn1(features)
        features = self.teacher.backbone.relu(features)
        features = self.teacher.backbone.maxpool(features)

        # Stage 1 (after layer1)
        features_low = self.teacher.backbone.layer1(features)

        # Stage 2 (output of layer2) - Used as the 'low' tap
        features_mid = self.teacher.backbone.layer2(features_low)

        # Stage 3 (output of layer3) - Used as the 'mid' tap
        features_high = self.teacher.backbone.layer3(features_mid)

        # Stage 4 (output of layer4) - Used as the 'high' tap
        features_final = self.teacher.backbone.layer4(features_high)

        # 3. Augment the standard output dictionary with the required feature taps
        output['features'] = {
            'out': output,
            'low': features_mid,
            'mid': features_high,
            'high': features_final,
        }
        return output
