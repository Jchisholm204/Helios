import torch
from typing import Dict, List


def calculate_recall_at_k(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:

    device = image_features.device

    # Normalize
    image_features = torch.nn.functional.normalize(image_features, dim=-1)
    text_features = torch.nn.functional.normalize(text_features,  dim=-1)

    sim_matrix = image_features @ text_features.t()
    N = sim_matrix.size(0)

    k_max = min(max(k_values), N)

    results = {}
    labels = torch.arange(N, device=device).unsqueeze(1)

    # ---- Image → Text ----
    _, topk_text_indices = torch.topk(sim_matrix, k=k_max, dim=1)

    for k in k_values:
        k = min(k, N)
        correct = (topk_text_indices[:, :k] == labels)
        results[f'I2T_R@{k}'] = correct.any(dim=1).float().mean().item()

    # ---- Text → Image ----
    _, topk_img_indices = torch.topk(sim_matrix.t(), k=k_max, dim=1)

    for k in k_values:
        k = min(k, N)
        correct = (topk_img_indices[:, :k] == labels)
        results[f'T2I_R@{k}'] = correct.any(dim=1).float().mean().item()

    return results


# --- Test Block ---
if __name__ == '__main__':
    # Test case: N=4, D=512
    N, D = 100, 512

    # Perfect alignment
    image_f = torch.randn(N, D)
    text_f = image_f.clone()

    perfect = calculate_recall_at_k(image_f, text_f)
    print("--- Perfect Alignment ---")
    print(perfect)

    # Mismatch
    text_f_shift = torch.roll(text_f, shifts=1, dims=0)
    mismatch = calculate_recall_at_k(image_f, text_f_shift)
    print("--- Mismatch ---")
    print(mismatch)
