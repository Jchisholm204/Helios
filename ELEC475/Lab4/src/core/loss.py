import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(temperature), requires_grad=False)

    def forward(self, image_features: torch.Tensor,
                text_features: torch.Tensor) -> torch.Tensor:
        batch_size = image_features.size(0)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 1. Compute Similarity Matrix (logits)
        # (N, D) @ (D, N) -> (N, N) matrix
        # S_ij = CosineSimilarity(Image_i, Text_j)
        logits = image_features @ text_features.t()

        # 2. Scale Logits by Temperature
        # NOTE: self.temperature must be a small fractional value like 0.07
        logits = logits / self.temperature

        # 3. Create Ground Truth Labels
        # The correct match for Image_i is Text_i (i.e., the diagonal index i).
        # Labels are simply indices 0, 1, ..., N-1
        labels = torch.arange(batch_size, device=logits.device)

        # 4. Compute Loss
        # a) Image-to-Text Loss (I2T)
        loss_i2t = F.cross_entropy(logits, labels)

        # b) Text-to-Image Loss (T2I)
        # This is equivalent to using the transpose of the logits matrix.
        loss_t2i = F.cross_entropy(logits.t(), labels)

        # 5. Return Symmetric Mean Loss
        total_loss = (loss_i2t + loss_t2i) / 2

        return total_loss


# --- Test Block ---
if __name__ == '__main__':
    # Test initialization and forward pass
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N, D = 256, 512  # Increase N for a more stable test average

    # Random, unaligned features (Used to represent the start of training)
    # The true starting loss should be -log(1/N). For N=256, -log(1/256) ≈ 5.54
    # To represent UNALIGNED features, we must generate I_emb and T_emb independently.

    # Generate image features from N(0, 1) and normalize
    image_features_random = torch.randn(N, D).to(test_device)
    image_features_random = image_features_random / \
        image_features_random.norm(dim=-1, keepdim=True)

    # Generate text features from a *different* random initialization and normalize
    text_features_random = torch.randn(N, D).to(test_device)
    text_features_random = text_features_random / \
        text_features_random.norm(dim=-1, keepdim=True)

    print(f"\n--- InfoNCE Loss Temperature Test (N={N}) ---")

    # Scenario 1: Tau = 1.0
    loss_fn_10 = InfoNCELoss(temperature=1.0).to(test_device)
    loss_10 = loss_fn_10(image_features_random, text_features_random)
    # The loss is around ln(N), which is 5.54. This is too large for training at tau=1.0.
    print(f"1. Loss with Tau=1.0: {loss_10.item()          :.4f} (UNSCALED, expected 5.5 - 5.8)")

    # Scenario 2: Tau = 0.07 (Recommended standard value)
    loss_fn_007 = InfoNCELoss(temperature=0.07).to(test_device)
    loss_007 = loss_fn_007(image_features_random, text_features_random)
    # Expected: The correct starting loss for N=256 is -log(1/256) ≈ 5.54
    print(f"2. Loss with Tau=0.07: {
          loss_007.item():.4f} (CORRECT START, expected 5.5 - 5.8)")

    # Scenario 3: Tau = 0.007 (Your attempted aggressive value)
    loss_fn_0007 = InfoNCELoss(temperature=0.007).to(test_device)
    loss_0007 = loss_fn_0007(image_features_random, text_features_random)
    # Expected: This scales the standard loss up considerably, making it too aggressive.
    print(f"3. Loss with Tau=0.007: {
          loss_0007.item():.4f} (TOO AGGRESSIVE, expected > 18.0)")

    # Assert that the loss with the recommended tau (0.07) is within the expected range
    if 4.0 < loss_007.item() < 6.5:
        print("\nPASS: Loss function produces correct starting loss for random features (Tau=0.07).")
    else:
        print(
            "\nFAIL: Starting loss is outside the expected range of [4.0, 6.5] at Tau=0.07.")
