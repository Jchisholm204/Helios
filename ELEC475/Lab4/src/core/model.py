import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Type
import math


class ResNet50Encoder(nn.Module):
    def __init__(self, weights=models.ResNet50_Weights.IMAGENET1K_V1,
                 freeze_backbone: bool = True):
        super(ResNet50Encoder, self).__init__()
        resnet = models.resnet50(weights=weights)
        # remove final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = resnet.fc.in_features

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        return x


class CLIPModel(nn.Module):
    def __init__(self, image_encoder: Type[ResNet50Encoder] = ResNet50Encoder,
                 projection_dim: int = 512, freeze_backbone: bool = True):
        super(CLIPModel, self).__init__()
        self.image_encoder = image_encoder(freeze_backbone=freeze_backbone)

        self.projection_head = nn.Sequential(  # two linear layers with GELU activation
            nn.Linear(self.image_encoder.output_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, images):
        image_features = self.image_encoder(images)
        image_embeddings = self.projection_head(image_features)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        return image_embeddings


if __name__ == '__main__':
    # Test initialization and forward pass
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Batch size 4, 3 channels, 224x224 input image
    dummy_input = torch.randn(4, 3, 224, 224).to(test_device)

    # Initialize the model (freezing the backbone is the default, training the projection head)
    model = CLIPModel(image_encoder=ResNet50Encoder,
                      projection_dim=512,
                      freeze_backbone=True).to(test_device)

    # Check parameter status
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print("\n--- CLIP Model Test ---")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (Projection Head only): {trainable_params:,}")

    # Test forward pass
    with torch.no_grad():
        output_features = model(dummy_input)

    print(f"Output features shape: {output_features.shape}")

    # Check L2 normalization (should be close to 1.0)
    norm = output_features.norm(dim=-1).mean().item()
    print(f"Average L2 norm of output features: {norm:.4f} (should be ≈ 1.0)")

    # Adjusted check range: The expected value is 1,312,256, so we check near that value.
    if trainable_params < 1_400_000 and trainable_params > 1_300_000:
        print(
            "PASS: Backbone appears frozen and Projection Head is trainable (1.3M params).")
    else:
        # Print a warning if it doesn't match the expectation, but allow continuation.
        print("FAIL: Check freezing logic. Trainable params count is unexpected.")
