import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE Loss (Contrastive Loss) for CLIP-style training.

    This loss encourages the similarity between matched (image, text) pairs 
    to be high, and the similarity between unmatched pairs to be low.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initializes the loss module.

        Args:
            temperature (float): Controls the contrastive force. Lower temperature 
                                 makes the distribution sharper, requiring higher 
                                 similarity for positive pairs.
        """
        super().__init__()
        self.temperature = temperature
        # Logits_per_image is the similarity matrix S (N x N)
        # The ground truth (correct pairs) are always on the diagonal.
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Computes the symmetric InfoNCE loss.

        Args:
            image_features (torch.Tensor): Image embeddings (N, D), already L2-normalized.
            text_features (torch.Tensor): Text embeddings (N, D), already L2-normalized.

        Returns:
            torch.Tensor: The mean loss of the batch.
        """
        batch_size = image_features.size(0)

        # 1. Compute Similarity Matrix (logits)
        # (N, D) @ (D, N) -> (N, N) matrix
        # S_ij = CosineSimilarity(Image_i, Text_j)
        logits = image_features @ text_features.T

        # 2. Scale Logits by Temperature
        logits = logits / self.temperature

        # 3. Create Ground Truth Labels
        # The correct match for Image_i is Text_i (i.e., the diagonal index i).
        # Labels are simply indices 0, 1, ..., N-1
        labels = torch.arange(batch_size, device=logits.device)

        # 4. Compute Loss
        # a) Image-to-Text Loss (I2T)
        # Treat the N rows as N classification problems, where the goal is to classify
        # the true text (labels[i]) given the image logits (logits[i, :]).
        loss_i2t = self.criterion(logits, labels)

        # b) Text-to-Image Loss (T2I)
        # Treat the N columns as N classification problems, where the goal is to classify
        # the true image (labels[j]) given the text logits (logits[:, j]).
        # This is equivalent to using the transpose of the logits matrix.
        loss_t2i = self.criterion(logits.T, labels)

        # 5. Return Symmetric Mean Loss
        total_loss = (loss_i2t + loss_t2i) / 2

        return total_loss


# --- Test Block ---
if __name__ == '__main__':
    # Test initialization and forward pass
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy features (4 items, 512 dimensions, L2 normalized)
    # We create two sets of features: F_I and F_T.
    N, D = 4, 512
    # Perfect alignment (Diagonal matrix is 1s, off-diagonal is 0s)
    image_features_perfect = torch.eye(N, D).to(test_device)
    text_features_perfect = torch.eye(N, D).to(test_device)

    # Random, unaligned features
    image_features_random = torch.randn(N, D).to(test_device)
    text_features_random = torch.randn(N, D).to(test_device)
    # L2 normalize the random features
    image_features_random = image_features_random / \
        image_features_random.norm(dim=-1, keepdim=True)
    text_features_random = text_features_random / \
        text_features_random.norm(dim=-1, keepdim=True)

    loss_fn = InfoNCELoss(temperature=1.0).to(test_device)

    print("\n--- InfoNCE Loss Test ---")

    # Test 1: Perfect Alignment (Loss should be low, close to 0)
    loss_perfect = loss_fn(image_features_perfect, text_features_perfect)
    print(f"Loss with PERFECT alignment: {
          loss_perfect.item():.4f} (Expected: Low)")

    # Test 2: Random Alignment (Loss should be high, relative to perfect)
    loss_random = loss_fn(image_features_random, text_features_random)
    print(f"Loss with RANDOM alignment: {
          loss_random.item():.4f} (Expected: High)")

    if loss_perfect.item() < loss_random.item():
        print("PASS: Loss function correctly penalizes misalignment.")
    else:
        print("FAIL: Loss function alignment check failed.")
