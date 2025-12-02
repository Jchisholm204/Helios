"""
Evaluate image<->text retrieval using the saved caption/image encodings.

Produces I2T and T2I Recall@{1,5,10}.
"""
import os
import torch
import torch.nn.functional as F
from typing import Dict, List

from dataset import Coco2014
from core.model import CLIPModel  # use your model for optional projection

ROOT = os.path.join(os.path.dirname(__file__), "..", "coco2014")
CAPTION_ENCODINGS = os.path.join(ROOT, "val_caption_encodings.pt")
IMAGE_ENCODINGS = os.path.join(ROOT, "val_image_encodings.pt")
MODEL_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "logs", "augmentation_20251130_105023", "best_model.pth")
DEVICE = "cpu"


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
                # try tensor-like
                try:
                    ik = int(k.item())
                except Exception:
                    continue
        # ensure list of tensors
        if isinstance(v, torch.Tensor):
            norm[ik] = [v.detach().cpu()]
        elif isinstance(v, (list, tuple)):
            norm[ik] = [x.detach().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in v]
        else:
            # fallback single item
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
            # if token dim >1, average across first dim (safe)
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

    text_mat = F.normalize(torch.stack(text_embs), dim=1).to(DEVICE)  # (M, D)
    image_mat = F.normalize(torch.stack(image_embs), dim=1).to(DEVICE)  # (N, D)
    return text_mat, image_mat, caption_to_image, image_ids


def recall_i2t(image_mat: torch.Tensor, text_mat: torch.Tensor, caption_to_image: List[int], image_ids: List[int], ks=(1, 5, 10)):
    # image -> text: for each image, check if any of the top-k retrieved captions correspond to that image
    sims = image_mat @ text_mat.T  # (N_images, M_captions)
    # Build caption index -> image id mapping
    cap2img = caption_to_image
    results = {}
    for k in ks:
        topk = sims.topk(k, dim=1).indices  # (N, k)
        hits = 0
        N = sims.size(0)
        for i, img_id in enumerate(image_ids):
            retrieved_cap_idxs = topk[i].tolist()
            # success if any retrieved caption maps to this image id
            ok = any(cap2img[cidx] == int(img_id) for cidx in retrieved_cap_idxs)
            if ok:
                hits += 1
        results[k] = hits / max(1, N)
    return results


def recall_t2i(image_mat: torch.Tensor, text_mat: torch.Tensor, caption_to_image: List[int], image_ids: List[int], ks=(1, 5, 10)):
    # text -> image: for each caption, check if any of top-k retrieved images corresponds to the caption's image
    sims = text_mat @ image_mat.T  # (M_captions, N_images)
    image_id_to_idx = {int(img_id): idx for idx, img_id in enumerate(image_ids)}
    M = sims.size(0)
    results = {}
    # filter valid captions (whose ground truth image exists in image_ids)
    gt_indices = []
    valid_caption_mask = []
    for cid_img in caption_to_image:
        if int(cid_img) in image_id_to_idx:
            gt_indices.append(image_id_to_idx[int(cid_img)])
            valid_caption_mask.append(True)
        else:
            gt_indices.append(-1)
            valid_caption_mask.append(False)
    gt_indices = torch.tensor(gt_indices, device=sims.device)
    valid_mask = torch.tensor(valid_caption_mask, dtype=torch.bool, device=sims.device)
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return {k: 0.0 for k in ks}
    for k in ks:
        topk = sims.topk(k, dim=1).indices  # (M, k)
        # only consider valid captions
        topk_valid = topk[valid_mask]
        gt_valid = gt_indices[valid_mask].unsqueeze(1)  # (V,1)
        correct = (topk_valid == gt_valid).any(dim=1).float().sum().item()
        results[k] = correct / valid_count
    return results


def project_to_clip_space(text_mat: torch.Tensor, image_mat: torch.Tensor, model_name: str = "openai/clip-vit-base-patch32"):
    """
    If text/image feature dims differ or are not in the CLIP joint space, use CLIPModel's
    text_projection and visual_projection to map them into the same projection_dim.
    Accepts projection stored as nn.Linear modules or tensors.
    Returns (text_mat_proj, image_mat_proj).
    """
    t_dim = text_mat.size(1)
    v_dim = image_mat.size(1)

    # Quick check: if dims already match, nothing to do
    if t_dim == v_dim:
        print("Text and Image embeddings have matching dimensions; no projection applied.")
        return text_mat, image_mat

    clip = CLIPModel.from_pretrained(model_name)
    clip.eval()

    text_proj_obj = getattr(clip, "text_projection", None)
    visual_proj_obj = getattr(clip, "visual_projection", None)
    if text_proj_obj is None or visual_proj_obj is None:
        # fallback: try attributes on submodules or named_parameters
        for n, p in clip.named_parameters():
            if "text_projection" in n and text_proj_obj is None:
                text_proj_obj = p
            if "visual_projection" in n and visual_proj_obj is None:
                visual_proj_obj = p

    def _proj_to_matrix(proj_obj):
        """
        Convert projection object (Tensor, Parameter, nn.Linear, module with .weight) into a
        plain Tensor of shape (in_dim, proj_dim) ready for X @ P where X has shape (N, in_dim).
        """
        if proj_obj is None:
            return None
        # Tensor / Parameter: assume shape (in_dim, proj_dim) or (proj_dim, in_dim)
        if isinstance(proj_obj, torch.Tensor):
            P = proj_obj
            # prefer shape where rows correspond to input dim
            if P.ndim == 2:
                if P.shape[0] == t_dim or P.shape[0] == v_dim:
                    return P
                # if it's (proj_dim, in_dim) transpose
                return P.T
            return P
        # nn.Linear or module with weight attribute
        if hasattr(proj_obj, "weight"):
            w = getattr(proj_obj, "weight")
            if isinstance(w, torch.Tensor) and w.ndim == 2:
                # nn.Linear.weight has shape (out_features, in_features) => transpose
                return w.T
        # last resort: try to convert to tensor
        try:
            P = torch.as_tensor(proj_obj)
            if P.ndim == 2:
                return P
        except Exception:
            return None
        return None

    text_proj = _proj_to_matrix(text_proj_obj)
    visual_proj = _proj_to_matrix(visual_proj_obj)

    if text_proj is None or visual_proj is None:
        raise RuntimeError("Could not extract usable projection matrices from CLIPModel.")

    # Move projection to same device as mats
    text_proj = text_proj.to(text_mat.device)
    visual_proj = visual_proj.to(image_mat.device)

    # Validate compatibility
    if text_proj.shape[0] != text_mat.size(1):
        raise RuntimeError(f"text_projection input dim {text_proj.shape[0]} != text_mat dim {text_mat.size(1)}")
    if visual_proj.shape[0] != image_mat.size(1):
        raise RuntimeError(f"visual_projection input dim {visual_proj.shape[0]} != image_mat dim {image_mat.size(1)}")

    with torch.no_grad():
        text_p = text_mat @ text_proj
        image_p = image_mat @ visual_proj

        text_p = F.normalize(text_p, dim=1)
        image_p = F.normalize(image_p, dim=1)
    return text_p, image_p


def project_with_custom_model_if_needed(text_mat: torch.Tensor, image_mat: torch.Tensor, weights_path: str = MODEL_WEIGHTS):
    """
    If dims mismatch, attempt to load your CLIPModel and apply its projection_head
    to whichever modality has input dim matching the model.image_encoder.output_dim.
    Otherwise raise informative error.
    """
    if text_mat.size(1) == image_mat.size(1):
        return text_mat, image_mat

    # load model
    model = CLIPModel()
    if not os.path.exists(weights_path):
        raise RuntimeError(f"Model weights not found: {weights_path}. Regenerate encodings with the model first.")
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
    # projection head is model.projection_head (nn.Sequential)
    with torch.no_grad():
        img_in_dim = model.image_encoder.output_dim
        # if image_mat is raw image encoder features -> project them
        if image_mat.size(1) == img_in_dim:
            proj_image = model.projection_head(image_mat)
            proj_image = F.normalize(proj_image, dim=1)
            # if text already in same dim as proj_image, done
            if proj_image.size(1) == text_mat.size(1):
                return text_mat, proj_image
            # if text differs, return projected image and leave caller to handle mismatch
            return text_mat, proj_image
        # if text_mat equals raw image encoder dims, project text (unlikely) - support anyway
        if text_mat.size(1) == img_in_dim:
            proj_text = model.projection_head(text_mat)
            proj_text = F.normalize(proj_text, dim=1)
            return proj_text, image_mat
    # no suitable projection path found
    raise RuntimeError(
        "Embeddings dimension mismatch and custom model projection could not be applied. "
        "Regenerate encodings using the model in src/core/model.py (image encodings should be projected)."
    )


def main():
    print("Loading encodings...")
    caps = load_encodings(CAPTION_ENCODINGS)
    imgs = load_encodings(IMAGE_ENCODINGS)

    # Use dataset to verify / load annotations (optional)
    ds = Coco2014(root=os.path.join(os.path.dirname(__file__), "..", "coco2014"), is_train=False)
    ds._init_self()

    print(f"Found {len(caps)} image keys in caption encodings, {len(imgs)} image keys in image encodings.")
    text_mat, image_mat, caption_to_image, image_ids = build_matrices(caps, imgs)

    # Ensure both modalities live in same space; try to use your model to project if needed
    if text_mat.size(1) != image_mat.size(1):
        print(f"Dim mismatch: text {text_mat.size(1)} vs image {image_mat.size(1)}. Trying custom model projection...")
        text_mat, image_mat = project_with_custom_model_if_needed(text_mat, image_mat)

    print("Computing recalls...")
    ks = (1, 5, 10)
    i2t = recall_i2t(image_mat, text_mat, caption_to_image, image_ids, ks)
    t2i = recall_t2i(image_mat, text_mat, caption_to_image, image_ids, ks)

    for k in ks:
        print(f"K={k} | I2T Recall@{k}: {i2t[k]*100:.2f}% | T2I Recall@{k}: {t2i[k]*100:.2f}%")

    # Optionally print some diagnostics
    total_captions = text_mat.size(0)
    total_images = image_mat.size(0)
    print(f"Total images evaluated: {total_images}; total captions: {total_captions}")


if __name__ == "__main__":
    main()