import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights


class AlexNetRegressor(nn.Module):
    """
    A regression model based on the pre-trained AlexNet architecture.

    The final classification layer (FC1000) is replaced with a linear layer
    to predict two continuous values (normalized x and y coordinates).
    """

    def __init__(self, output_size=2, freeze_backbone=True):
        """
        Initializes the AlexNetRegressor.

        Args:
            output_size (int): The number of output values (default is 2 for [x, y]).
            freeze_backbone (bool): If True, freezes the weights of the feature 
                                    extractor (convolutional layers).
        """
        super(AlexNetRegressor, self).__init__()

        # Load the pre-trained AlexNet model with ImageNet weights
        self.alexnet = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        # --- Adaptation for Regression ---

        # The AlexNet classifier is a sequential block (nn.Sequential).
        # We target the last layer in that block, which is the FC1000 layer.
        classifier = self.alexnet.classifier

        # Get the input features of the existing final layer (it should be 4096)
        num_ftrs = classifier[6].in_features

        # Replace the final layer with a new linear layer for 2-output regression
        classifier[6] = nn.Linear(num_ftrs, output_size)

        # --- Optional Freezing of Feature Extractor ---

        if freeze_backbone:
            # Freeze weights in the convolutional layers ('features' part)
            for param in self.alexnet.features.parameters():
                param.requires_grad = False

            # The classifier layers (FC layers) are typically unfrozen
            # to allow the model to learn the new regression task quickly.
            # Only the final layer was modified, but the FC layers before it
            # are usually fine-tuned.

        print(f"AlexNetRegressor initialized. Output layer set to: {
              self.alexnet.classifier[6]}")

    def forward(self, x):
        """Passes the input through the modified AlexNet."""
        return self.alexnet(x)


if __name__ == '__main__':
    # Example usage and sanity check:
    model = AlexNetRegressor(output_size=2)

    # Check if the output layer size is correct
    print(f"\nFinal output layer: {model.alexnet.classifier[6]}")

    # Check output with a dummy tensor (Batch size=4, 3 channels, 227x227)
    dummy_input = torch.randn(4, 3, 227, 227)
    output = model(dummy_input)

    print(f"Dummy input size: {dummy_input.shape}")
    print(f"Model output size: {output.shape} (Expected: 4, 2)")
