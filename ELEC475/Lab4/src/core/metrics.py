import torch
from typing import Dict, List


def calculate_recall_at_k(image_features: torch.Tensor, text_features: torch.Tensor, k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Computes Recall@K for Image-to-Text (I2T) and Text-to-Image (T2I) retrieval.

    Recall@K answers: "Is the correct target found among the top K predictions?"

    Args:
        image_features (torch.Tensor): Image embeddings (N, D), assumed L2-normalized.
        text_features (torch.Tensor): Text embeddings (N, D), assumed L2-normalized.
        k_values (list): List of K values to check (e.g., [1, 5, 10]).

    Returns:
        Dict[str, float]: Dictionary of all Recall@K results (e.g., {'R@1_I2T': 0.5}).
    """

    # Ensure inputs are on the same device
    device = image_features.device

    # 1. Compute the Similarity Matrix (N x N)
    # Cosine Similarity is equivalent to the dot product of L2-normalized vectors.
    sim_matrix = image_features @ text_features.T

    N = sim_matrix.size(0)
    results = {}

    # Ground truth: The correct match index is always on the diagonal (0, 1, ..., N-1)
    ground_truth = torch.arange(N).to(device)

    # --- 2. Image to Text (I2T) Retrieval ---
    # Goal: For each image, find the rank of its true caption among all captions.

    # Sort the similarity matrix rows to get the indices of captions ranked by similarity
    # i2t_ranks[i, j] is the index of the j-th most similar caption for image i.
    i2t_ranks = sim_matrix.argsort(descending=True)

    for k in k_values:
        # Check if the ground truth index is present in the top-K predicted indices
        # i2t_ranks[:, :k] are the top K indices for every image.
        # ground_truth.unsqueeze(1) repeats the true index for comparison.
        i2t_hits = torch.any(i2t_ranks[:, :k] ==
                             ground_truth.unsqueeze(1), dim=1)

        # Recall is the average of hits (True = 1.0, False = 0.0)
        recall = i2t_hits.float().mean().item()
        results[f'R@{k}_I2T'] = recall

    # --- 3. Text to Image (T2I) Retrieval ---
    # Goal: For each caption, find the rank of its true image among all images.

    # We transpose the matrix and sort the columns (which are now rows)
    t2i_ranks = sim_matrix.T.argsort(descending=True)

    for k in k_values:
        t2i_hits = torch.any(t2i_ranks[:, :k] ==
                             ground_truth.unsqueeze(1), dim=1)
        recall = t2i_hits.float().mean().item()
        results[f'R@{k}_T2I'] = recall

    return results


# --- Test Block ---
if __name__ == '__main__':
    # Test case: N=4, D=512
    N, D = 4, 512
    # Create perfect alignment (diagonal is high, rest is low)
    sim = torch.eye(N) * 5.0 + torch.randn(N, N) * 0.1
    # Create features from this similarity (This is a simplified test, usually features are used)
    # Since the function takes features, we create dummy normalized features:
    image_f = torch.eye(N, D)
    text_f = torch.eye(N, D)

    # Test perfect match (should get 100% recall)
    perfect_results = calculate_recall_at_k(image_f, text_f)
    print("--- Recall@K Test (Perfect Alignment) ---")
    print(perfect_results)  # Should be R@1 = 1.0

    # Test case 2: Total mismatch (swapped features)
    text_f_mismatch = torch.cat(
        [text_f[1:], text_f[0].unsqueeze(0)], dim=0)  # Shifted by 1
    mismatch_results = calculate_recall_at_k(image_f, text_f_mismatch)
    print("\n--- Recall@K Test (Mismatch) ---")
    # Should be R@1 = 0.0, R@5/10 might be 1.0 if N is small
    print(mismatch_results)
