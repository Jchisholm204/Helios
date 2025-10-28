import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class VGG16Regressor(nn.Module):
    """
    A regression model based on the pre-trained VGG16 architecture.

    The final classification layer (FC1000) is replaced with a linear layer
    to predict two continuous values (normalized x and y coordinates).
    """

    def __init__(self, output_size=2, freeze_backbone=True):
        """
        Initializes the VGG16Regressor.

        Args:
            output_size (int): The number of output values (default is 2 for [x, y]).
            freeze_backbone (bool): If True, freezes the weights of the feature 
                                    extractor (convolutional layers).
        """
        super(VGG16Regressor, self).__init__()

        # Load the pre-trained VGG16 model with ImageNet weights
        # We use VGG16_Weights.IMAGENET1K_V1 for standard pre-trained weights
        self.vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        # --- Adaptation for Regression ---

        # The VGG16 classifier is a sequential block (nn.Sequential).
        # It consists of three fully connected layers: FC4096 -> FC4096 -> FC1000.
        # We target the last layer in that block, which is index [6].
        classifier = self.vgg16.classifier

        # Get the input features of the existing final layer (it should be 4096)
        num_ftrs = classifier[6].in_features

        # Replace the final layer with a new linear layer for 2-output regression
        # VGG's classifier is a nn.Sequential, so we directly modify the 6th element.
        classifier[6] = nn.Linear(num_ftrs, output_size)

        # --- Optional Freezing of Feature Extractor ---

        if freeze_backbone:
            # Freeze weights in the convolutional layers ('features' part)
            for param in self.vgg16.features.parameters():
                param.requires_grad = False

        print(f"VGG16Regressor initialized. Output layer set to: {
              self.vgg16.classifier[6]}")

    def forward(self, x):
        """Passes the input through the modified VGG16."""
        return self.vgg16(x)


if __name__ == '__main__':
    # Example usage and sanity check:
    model = VGG16Regressor(output_size=2)

    # Check if the output layer size is correct
    print(f"\nFinal output layer: {model.vgg16.classifier[6]}")

    # Check output with a dummy tensor (Batch size=4, 3 channels, 224x224)
    # VGG16 typically expects 224x224, but 227x227 (AlexNet size) also works due to pooling
    dummy_input = torch.randn(4, 3, 227, 227)
    output = model(dummy_input)

    print(f"Dummy input size: {dummy_input.shape}")
    print(f"Model output size: {output.shape} (Expected: 4, 2)")
