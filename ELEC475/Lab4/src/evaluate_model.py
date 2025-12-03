"""
Evaluate image<->text retrieval using the saved caption/image encodings.

Produces I2T and T2I Recall@{1,5,10} and provides simple visualization & classification helpers:
 - retrieve_and_show(query, topk=5) : display top-k images for a text query
 - classify_image_by_classes(img_id, classes): score/classify an image given class strings

Uses:
 - saved encodings: coco2014/val_caption_encodings.pt and val_image_encodings.pt
 - transformers CLIPTextModel/Tokenizer to encode ad-hoc queries/classes
 - your custom src/core/model.CLIPModel only as fallback projection (not required if encodings are already projected)
"""
import os
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

from dataset import Coco2014
# used only for fallback projection if needed (kept for compatibility)
from core.model import CLIPModel

# for on-the-fly text encoding and visualization
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import matplotlib.pyplot as plt

MODEL = "basemodel"
ROOT = os.path.join(os.path.dirname(__file__), "..", "coco2014")
ROOT_PTH = os.path.join(f"./logs", MODEL)
CAPTION_ENCODINGS = os.path.join(ROOT, "val_caption_encodings.pt")
IMAGE_ENCODINGS = os.path.join(ROOT_PTH, "val_image_encodings.pt")
MODEL_WEIGHTS = os.path.join(ROOT_PTH, "final_model.pth")

# encoding model name used by caption_encoder.py
ENCODER_MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cpu"  # keep CPU-friendly by default


def load_encodings(path: str) -> Dict[int, List[torch.Tensor]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    enc = torch.load(path)
    # Keys may be strings / tensors; normalize to int -> list[tensor]
    norm = {}
    for k, v in enc.items():
        try:
            ik = int(k)
        except Exception:
            try:
                ik = int(str(k))
            except Exception:
                try:
                    ik = int(k.item())
                except Exception:
                    continue
        # ensure list of tensors, on CPU
        if isinstance(v, torch.Tensor):
            norm[ik] = [v.detach().cpu()]
        elif isinstance(v, (list, tuple)):
            norm[ik] = [x.detach().cpu() if isinstance(x, torch.Tensor)
                        else torch.as_tensor(x) for x in v]
        else:
            norm[ik] = [torch.as_tensor(v)]
    return norm


def build_matrices(caps: Dict[int, List[torch.Tensor]], imgs: Dict[int, List[torch.Tensor]]):
    # Build text embeddings (one entry per caption variant) and mapping caption_idx -> image_id
    caption_to_image: List[int] = []
    text_embs = []
    for img_id in sorted(caps.keys()):
        variants = caps[img_id]
        for v in variants:
            t = v.float()
            if t.ndim > 1:
                t = t.mean(dim=0)
            text_embs.append(t)
            caption_to_image.append(int(img_id))
    if len(text_embs) == 0:
        raise RuntimeError("No caption embeddings found.")

    # Build image embeddings (one per image). Average variants per image.
    image_ids = sorted(imgs.keys())
    image_embs = []
    for img_id in image_ids:
        variants = imgs[img_id]
        vs = [v.float() for v in variants]
        if len(vs) == 0:
            continue
        stacked = torch.stack(vs)
        mean = stacked.mean(dim=0)
        image_embs.append(mean)

    if len(image_embs) == 0:
        raise RuntimeError("No image embeddings found.")

    text_mat = F.normalize(torch.stack(text_embs),
                           dim=1).to(DEVICE)  # (M, D_text)
    image_mat = F.normalize(torch.stack(image_embs),
                            dim=1).to(DEVICE)  # (N, D_image)
    return text_mat, image_mat, caption_to_image, image_ids


def recall_i2t(image_mat: torch.Tensor, text_mat: torch.Tensor, caption_to_image: List[int], image_ids: List[int], ks=(1, 5, 10), chunk_size: int = 512):
    """
    Image->Text Recall@k computed in image chunks to avoid allocating full (N x M) sims.
    Returns:
      results: dict k -> recall (float)
      hits_per_k: dict k -> list[int] (0/1) per image (length = num_images)
    """
    device = image_mat.device
    M = text_mat.size(0)
    N = image_mat.size(0)
    max_k = max(ks)
    hits_count = {k: 0 for k in ks}
    hits_per_k = {k: [0] * N for k in ks}
    cap2img = caption_to_image

    for i0 in range(0, N, chunk_size):
        i1 = min(N, i0 + chunk_size)
        imgs_chunk = image_mat[i0:i1]                 # (B, D)
        sims = imgs_chunk @ text_mat.T               # (B, M)
        topk = sims.topk(max_k, dim=1).indices       # (B, max_k)
        topk = topk.cpu().tolist()
        for local_idx, retrieved in enumerate(topk):
            img_global_idx = i0 + local_idx
            img_id = int(image_ids[img_global_idx])
            for k in ks:
                ok = any(cap2img[cidx] == img_id for cidx in retrieved[:k])
                hits_per_k[k][img_global_idx] = 1 if ok else 0
                if ok:
                    hits_count[k] += 1

    results = {k: hits_count[k] / max(1, N) for k in ks}
    return results, hits_per_k


def recall_t2i(image_mat: torch.Tensor, text_mat: torch.Tensor, caption_to_image: List[int], image_ids: List[int], ks=(1, 5, 10), chunk_size: int = 2048):
    """
    Text->Image Recall@k computed in caption chunks to avoid allocating full (M x N) sims.
    Returns:
      results: dict k -> recall (float)
      hits_per_k: dict k -> list[int] (0/1) per valid caption (length = num_valid_captions)
    """
    device = image_mat.device
    N = image_mat.size(0)
    M = text_mat.size(0)
    max_k = max(ks)
    image_id_to_idx = {int(img_id): idx for idx,
                       img_id in enumerate(image_ids)}

    # Build ground-truth indices and valid mask
    gt_indices = []
    valid_mask_list = []
    for cid_img in caption_to_image:
        if int(cid_img) in image_id_to_idx:
            gt_indices.append(image_id_to_idx[int(cid_img)])
            valid_mask_list.append(True)
        else:
            gt_indices.append(-1)
            valid_mask_list.append(False)
    gt_indices = torch.tensor(gt_indices, device=device)
    valid_mask = torch.tensor(valid_mask_list, dtype=torch.bool, device=device)
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return {k: 0.0 for k in ks}, {k: [] for k in ks}

    hits_count = {k: 0 for k in ks}
    # we'll accumulate per-valid-caption hits in lists (only valid captions)
    hits_per_k_valid = {k: [] for k in ks}

    for c0 in range(0, M, chunk_size):
        c1 = min(M, c0 + chunk_size)
        text_chunk = text_mat[c0:c1]                 # (B, D)
        sims = text_chunk @ image_mat.T              # (B, N)
        topk = sims.topk(max_k, dim=1).indices       # (B, max_k)

        # only consider valid captions in this chunk
        chunk_gt = gt_indices[c0:c1].to(topk.device)        # (B,)
        chunk_valid = valid_mask[c0:c1].to(topk.device)     # (B,)
        if chunk_valid.sum().item() == 0:
            continue
        topk_valid = topk[chunk_valid]                       # (V, max_k)
        gt_valid = chunk_gt[chunk_valid].unsqueeze(1)        # (V,1)

        # For each k compute per-caption hit
        for k in ks:
            correct_mask = (topk_valid[:, :k] == gt_valid).any(dim=1)  # (V,)
            correct_list = correct_mask.cpu().int().tolist()
            hits_per_k_valid[k].extend(correct_list)
            hits_count[k] += int(correct_mask.sum().item())

    results = {k: hits_count[k] / max(1, valid_count) for k in ks}
    return results, hits_per_k_valid


def _compute_stats_from_hits(hits_list: List[int]):
    """
    Given a list of 0/1 ints, return dict with min,max,mean,std (as floats 0..100).
    If list is empty, return zeros.
    """
    if len(hits_list) == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    t = torch.tensor(hits_list, dtype=torch.float32)
    mn = float(t.min().item()) * 100.0
    mx = float(t.max().item()) * 100.0
    mean = float(t.mean().item()) * 100.0
    std = float(t.std(unbiased=False).item()) * 100.0
    return {"min": mn, "max": mx, "mean": mean, "std": std}


# ---------------------------
# Text encoding and utilities
# ---------------------------
_tokenizer = None
_text_model = None


def get_text_encoder():
    global _tokenizer, _text_model
    if _tokenizer is None or _text_model is None:
        _tokenizer = CLIPTokenizer.from_pretrained(ENCODER_MODEL_NAME)
        _text_model = CLIPTextModel.from_pretrained(
            ENCODER_MODEL_NAME).to(DEVICE).eval()
    return _tokenizer, _text_model


def encode_texts(texts: List[str]) -> torch.Tensor:
    """
    Tokenize and encode texts -> normalized (num_texts, D) tensor on DEVICE.
    """
    tokenizer, text_model = get_text_encoder()
    inputs = tokenizer(texts, return_tensors="pt",
                       padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = text_model(**inputs)
        emb = out.pooler_output  # (B, D)
        emb = F.normalize(emb, dim=1)
    return emb.cpu()


# ---------------------------
# Visualization / classification
# ---------------------------
COCO_VAL_IMAGES = os.path.join(ROOT, "images", "val2014")


def _load_image_by_id(img_id: int) -> Image.Image:
    fname = f"COCO_val2014_{int(img_id):012d}.jpg"
    path = os.path.join(COCO_VAL_IMAGES, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def retrieve_and_show(query: str, image_mat: torch.Tensor, image_ids: List[int], topk: int = 5, figsize: Tuple[int, int] = (15, 6)):
    """
    Encode 'query' (using CLIPTextModel) and display topk retrieved images (matplotlib).
    """
    q_emb = encode_texts([f"a photo of {query}"])[
        0].to(image_mat.device)  # (D,)
    sims = (image_mat @ q_emb.unsqueeze(1)).squeeze(1)  # (N,)
    vals, idxs = sims.topk(topk)
    ids = [image_ids[i] for i in idxs.tolist()]

    # Plot
    plt.figure(figsize=figsize)
    for i, iid in enumerate(ids):
        try:
            img = _load_image_by_id(iid)
        except FileNotFoundError:
            continue
        ax = plt.subplot(1, topk, i + 1)
        plt.imshow(img)
        plt.axis("off")
        ax.set_title(f"id:{iid}\nscore:{vals[i]:.3f}")
    plt.suptitle(f"Top-{topk} retrievals for query: '{query}'")
    plt.tight_layout()
    plt.show()


def classify_image_by_classes(img_id: int, classes: List[str], image_mat: torch.Tensor, image_ids: List[int]) -> List[Tuple[str, float]]:
    """
    Given an image id and list of class strings (e.g., ['a person','an animal','a landscape']),
    compute similarity scores and return sorted list of (class, score).
    """
    # find image index
    try:
        idx = image_ids.index(int(img_id))
    except ValueError:
        raise RuntimeError(f"Image id {img_id} not found in loaded image ids.")

    img_emb = image_mat[idx].unsqueeze(0)  # (1, D)
    # Encode class strings (prefix with 'a photo of ' to be consistent)
    prefixed = [f"a photo of {c}" for c in classes]
    class_embs = encode_texts(prefixed).to(image_mat.device)  # (C, D)
    sims = (class_embs @ img_emb.T).squeeze(1)  # (C,)
    sims = sims.cpu()
    pairs = list(zip(classes, sims.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


# ---------------------------
# Optional: attempt to align dims using custom model (if needed)
# ---------------------------
def project_with_custom_model_if_needed(text_mat: torch.Tensor, image_mat: torch.Tensor, weights_path: str = MODEL_WEIGHTS):
    """
    If dims mismatch, attempt to load your CLIPModel and apply its projection_head
    to whichever modality has input dim matching the model.image_encoder.output_dim.
    Otherwise raise informative error.
    """
    if text_mat.size(1) == image_mat.size(1):
        return text_mat, image_mat

    model = CLIPModel()
    if not os.path.exists(weights_path):
        raise RuntimeError(f"Model weights not found: {
                           weights_path}. Regenerate encodings with the model first.")
    sd = torch.load(weights_path, map_location=DEVICE)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    try:
        model.load_state_dict(sd, strict=False)
    except Exception:
        new_sd = {}
        for k, v in sd.items():
            nk = k.replace("module.", "")
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)
    model.eval()

    with torch.no_grad():
        # model.image_encoder.output_dim should reflect raw image encoder output dim
        img_in_dim = getattr(model.image_encoder, "output_dim", None)
        # projection head should be callable on tensors
        proj = getattr(model, "projection_head", None)
        if img_in_dim is None or proj is None:
            raise RuntimeError(
                "Custom model does not expose expected attributes (image_encoder.output_dim / projection_head).")

        # If image_mat are raw image encoder features, project them
        if image_mat.size(1) == img_in_dim:
            proj_image = proj(image_mat)
            proj_image = F.normalize(proj_image, dim=1)
            return text_mat, proj_image

        # If text_mat matches image encoder dim (unlikely), project text
        if text_mat.size(1) == img_in_dim:
            proj_text = proj(text_mat)
            proj_text = F.normalize(proj_text, dim=1)
            return proj_text, image_mat

    raise RuntimeError(
        "Embeddings dimension mismatch and custom model projection could not be applied. Regenerate encodings with your model so both modalities share the same projected dim.")


def main():
    print("Loading encodings...")
    caps = load_encodings(CAPTION_ENCODINGS)
    imgs = load_encodings(IMAGE_ENCODINGS)

    # Use dataset to verify / load annotations (optional)
    ds = Coco2014(root=os.path.join(os.path.dirname(
        __file__), "..", "coco2014"), is_train=False)
    ds._init_self()

    print(f"Found {len(caps)} image keys in caption encodings, {
          len(imgs)} image keys in image encodings.")
    text_mat, image_mat, caption_to_image, image_ids = build_matrices(
        caps, imgs)

    # Align dims if mismatch using your model projection (attempt)
    if text_mat.size(1) != image_mat.size(1):
        print(f"Dim mismatch: text {text_mat.size(1)} vs image {
              image_mat.size(1)}. Trying custom model projection...")
        text_mat, image_mat = project_with_custom_model_if_needed(
            text_mat, image_mat)

    print("Computing recalls and statistics...")
    do_recall = True
    if do_recall:
        ks = (1, 5, 10)
        i2t_res, i2t_hits = recall_i2t(
            image_mat, text_mat, caption_to_image, image_ids, ks)
        t2i_res, t2i_hits = recall_t2i(
            image_mat, text_mat, caption_to_image, image_ids, ks)

        for k in ks:
            i2t_stat = _compute_stats_from_hits(i2t_hits[k])
            t2i_stat = _compute_stats_from_hits(t2i_hits[k])

            print(f"K={k} | I2T Recall@{k}: {i2t_res[k]*100:.2f}% | "
                  f"min={i2t_stat['min']:.2f}% max={i2t_stat['max']:.2f}% mean={i2t_stat['mean']:.2f}% std={i2t_stat['std']:.2f}%")

            print(f"      T2I Recall@{k}: {t2i_res[k]*100:.2f}% | "
                  f"min={t2i_stat['min']:.2f}% max={t2i_stat['max']:.2f}% mean={t2i_stat['mean']:.2f}% std={t2i_stat['std']:.2f}%")

    print(f"Total images evaluated: {image_mat.size(
        0)}; total captions: {text_mat.size(0)}")

    # -------------------------
    # Example interactive usage
    # -------------------------
    try:
        # Example 1: retrieve top-5 images for text query 'sport'
        print("\nExample retrieval for query: 'sport' (shows plot window)...")
        retrieve_and_show("sport", image_mat, image_ids, topk=5)

        # Example 2: classify an example image using class candidates
        example_img_id = image_ids[0]
        classes = ["a dog", "a shoe", "a basket"]
        scores = classify_image_by_classes(
            example_img_id, classes, image_mat, image_ids)
        print(f"\nClassification for image id {example_img_id}:")
        for cls, score in scores:
            print(f"  {cls}: {score:.4f}")
    except Exception as e:
        # keep main evaluation robust; print error but don't crash silently
        print(f"Visualization/classification example failed: {e}")


if __name__ == "__main__":
    main()
