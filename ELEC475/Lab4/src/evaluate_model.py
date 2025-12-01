import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision.utils import make_grid
import torchvision.transforms as T

from core.model import CLIPModel
from dataloader import Coco2014Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 150
OUT_DIR = "logs/retrievals"
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()
def compute_text_matrix(val_dataset):
    """
    Build:
      caption_ids: list[int]  -- annotation/caption ids (ints)
      text_embs:  tensor (M, D)
      caption_to_image: dict[int -> int] map caption id -> image id

    Robustly handles a few different annotation layouts:
      - image_to_ann_ids: {image_id: [ann_id, ...]}
      - annotations: list[dict] with 'id' and 'image_id'
      - annotations: list[int] (annotation ids) + anns / ann dict elsewhere
      - direct ann_id -> image_id maps under various names
    Also normalizes encoding keys to int and tolerates single-tensor / list-of-tensors variants.
    """
    val_dataset._init_self()
    encs_raw = getattr(val_dataset, "encodings", None)
    if encs_raw is None:
        raise RuntimeError("val_dataset has no encodings loaded (encodings_path?).")

    encs = {}
    for k, v in encs_raw.items():
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
                    # skip keys not coerced
                    continue
        encs[ik] = v

    caption_to_image = {}

    anns = val_dataset.annotations
    ann_map = anns.get("image_to_ann_ids") or anns.get("image_id_to_ann_ids") or anns.get("img_to_ann_ids")
    if ann_map:
        for img_id, ann_ids in ann_map.items():
            for aid in ann_ids:
                try:
                    caption_to_image[int(aid)] = int(img_id)
                except Exception:
                    try:
                        caption_to_image[int(str(aid))] = int(str(img_id))
                    except Exception:
                        pass
    else:
        annotations_entry = anns.get("annotations")
        if annotations_entry:
            if isinstance(annotations_entry, list) and len(annotations_entry) > 0:
                first = annotations_entry[0]
                if isinstance(first, dict):
                    for ann in annotations_entry:
                        aid = ann.get("id")
                        img = ann.get("image_id")
                        if aid is not None and img is not None:
                            try:
                                caption_to_image[int(aid)] = int(img)
                            except Exception:
                                try:
                                    caption_to_image[int(str(aid))] = int(str(img))
                                except Exception:
                                    pass
                else:
                    # list of ints (annotation ids), try to find ann dict mapping elsewhere
                    ann_dict_candidates = (
                        anns.get("anns")
                        or anns.get("annotations_by_id")
                        or anns.get("ann_id_to_ann")
                        or anns.get("ann_map")
                    )
                    if isinstance(ann_dict_candidates, dict):
                        for aid in annotations_entry:
                            try:
                                annobj = ann_dict_candidates.get(aid) if aid in ann_dict_candidates else ann_dict_candidates.get(str(aid))
                            except Exception:
                                annobj = None
                            if isinstance(annobj, dict):
                                img = annobj.get("image_id")
                                if img is not None:
                                    try:
                                        caption_to_image[int(aid)] = int(img)
                                    except Exception:
                                        try:
                                            caption_to_image[int(str(aid))] = int(str(img))
                                        except Exception:
                                            pass
            elif isinstance(annotations_entry, dict):
                # maybe mapping id relates to ann dict
                for aid, ann in annotations_entry.items():
                    if isinstance(ann, dict):
                        img = ann.get("image_id")
                        if img is not None:
                            try:
                                caption_to_image[int(aid)] = int(img)
                            except Exception:
                                try:
                                    caption_to_image[int(str(aid))] = int(str(img))
                                except Exception:
                                    pass

        if not caption_to_image:
            direct_maps = (
                anns.get("ann_id_to_image")
                or anns.get("ann_id_to_image_id")
                or anns.get("ann_to_image")
                or anns.get("ann_id_to_img")
                or {}
            )
            if isinstance(direct_maps, dict):
                for k, v in direct_maps.items():
                    try:
                        caption_to_image[int(k)] = int(v)
                    except Exception:
                        try:
                            caption_to_image[int(str(k))] = int(str(v))
                        except Exception:
                            pass

    caption_ids = sorted(encs.keys())
    text_embs = []
    for cid in caption_ids:
        variants = encs[cid]
        try:
            if isinstance(variants, torch.Tensor):
                mean_emb = variants.mean(dim=0)
            else:
                stacked = torch.stack(variants)
                mean_emb = stacked.mean(dim=0)
        except Exception:
            try:
                stacked = torch.stack([torch.as_tensor(x) for x in variants])
                mean_emb = stacked.mean(dim=0)
            except Exception:
                continue
        text_embs.append(mean_emb)

    if len(text_embs) == 0:
        raise RuntimeError("No text embeddings could be constructed from val_dataset.encodings")

    text_embs = torch.stack(text_embs)         # (M, D)
    text_embs = F.normalize(text_embs, dim=1)
    return caption_ids, text_embs.to(DEVICE), caption_to_image


@torch.no_grad()
def compute_image_embeddings(model, val_loader, val_dataset):
    """
    Returns:
      image_embs: tensor (N_images, D)
      true_caption_ids: list[int]  -- chosen annotation id per image (length N_images)
      image_ids: list[int]         -- image ids in the same order
    """
    image_embs = []
    true_caption_ids = []
    image_ids = []

    val_dataset._init_self()
    ann_map = val_dataset.annotations.get('image_to_ann_ids', {})
    encs_raw = getattr(val_dataset, "encodings", None)
    encs_int_keys = set(int(k) for k in encs_raw.keys()) if encs_raw is not None else set()

    for images, img_id_batch in tqdm(val_loader, desc="Images"):
        # images: [B, 3, H, W]
        images = images.to(DEVICE)

        feats = model(images)
        feats = F.normalize(feats, dim=1)
        image_embs.append(feats.cpu())

        for iid in img_id_batch:
            img_id = int(iid)
            image_ids.append(img_id)

            chosen = None
            ann_candidates = ann_map.get(img_id, [])
            if ann_candidates:
                # prefer candidate that has an encoding
                for aid in ann_candidates:
                    if int(aid) in encs_int_keys:
                        chosen = int(aid)
                        break
                if chosen is None:
                    chosen = int(ann_candidates[0])
            else:
                chosen = img_id if img_id in encs_int_keys else img_id

            true_caption_ids.append(chosen)

    return torch.cat(image_embs), true_caption_ids, image_ids


def recall_image_to_text(image_emb, text_emb, true_caption_ids, caption_ids, k):
    """
    Image -> Text recall@k
    image_emb: (N, D)
    text_emb:  (M, D)
    true_caption_ids: len N
    caption_ids: len M
    """
    sims = image_emb @ text_emb.T  # (N, M)
    cid_to_index = {cid: i for i, cid in enumerate(caption_ids)}
    # map ground truth caption ids to text matrix indices
    gt_indices = torch.tensor([cid_to_index[int(c)] for c in true_caption_ids], device=sims.device)

    topk = sims.topk(k, dim=1).indices        # (N, k)
    correct = (topk == gt_indices.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


def recall_text_to_image(image_emb, text_emb, caption_ids, caption_to_image, image_ids, k):
    """
    Text -> Image recall@k
    image_emb: (N, D)
    text_emb:  (M, D)
    caption_ids: list length M (caption ids in same order as text_emb)
    caption_to_image: map caption id -> image id
    image_ids: list length N (image ids in same order as image_emb)
    """
    sims_T = text_emb @ image_emb.T  # (M, N)
    image_id_to_index = {int(iid): idx for idx, iid in enumerate(image_ids)}
    # build ground truth image index per caption (M length)
    gt_img_indices = []
    for cid in caption_ids:
        img_id = caption_to_image.get(int(cid))
        if img_id is None:
            # if no mapping, mark as -1 (miss)
            gt_img_indices.append(-1)
        else:
            # try to map to index
            try:
                gt_idx = image_id_to_index.get(int(img_id), -1)
            except Exception:
                # try str then int
                try:
                    gt_idx = image_id_to_index.get(int(str(img_id)), -1)
                except Exception:
                    gt_idx = -1
            gt_img_indices.append(gt_idx)
    gt_img_indices = torch.tensor(gt_img_indices, device=sims_T.device)

    topk_imgs = sims_T.topk(k, dim=1).indices  # (M, k)
    valid_mask = gt_img_indices >= 0
    if valid_mask.sum().item() == 0:
        return 0.0
    relevant_topk = topk_imgs[valid_mask]
    relevant_gt = gt_img_indices[valid_mask].unsqueeze(1)
    correct = (relevant_topk == relevant_gt).any(dim=1).float()
    return correct.mean().item()


def save_image_grid(image_paths, out_path, titles=None, max_size=(224, 224)):
    imgs = []
    for p in image_paths:
        try:
            im = Image.open(p).convert("RGB")
            im = im.resize(max_size)
            imgs.append(T.ToTensor()(im))
        except Exception:
            # placeholder gray image
            imgs.append(torch.ones(3, max_size[1], max_size[0]) * 0.5)
    grid = make_grid(imgs, nrow=len(imgs), normalize=True, scale_each=True)
    ndarr = (grid.permute(1, 2, 0).numpy() * 255).astype('uint8')
    Image.fromarray(ndarr).save(out_path)


def retrieve_images_for_text_query(query, caption_ids, text_matrix, val_dataset, image_embs, image_ids, topk=5):
    """
    Find captions that contain the query (case-insensitive), average their embeddings
    and retrieve top-k images by cosine similarity.
    """
    ann_id_to_caption = val_dataset.annotations.get('ann_id_to_caption', {})
    matches = [cid for cid, txt in ann_id_to_caption.items() if query.lower() in txt.lower()]
    if not matches:
        print("No captions matched the query string.")
        return []

    cid_to_index = {cid: i for i, cid in enumerate(caption_ids)}
    idxs = [cid_to_index[int(m)] for m in matches if int(m) in cid_to_index]
    if not idxs:
        print("No matched captions have embeddings.")
        return []

    q_emb = text_matrix[idxs].mean(dim=0, keepdim=True)  # (1, D)
    sims = q_emb @ image_embs.T.to(DEVICE)               # (1, N)
    topk_idx = sims.topk(topk, dim=1).indices.squeeze(0).tolist()
    top_image_ids = [image_ids[i] for i in topk_idx]

    paths = []
    for iid in top_image_ids:
        meta = val_dataset.annotations['images'].get(iid, {})
        fname = meta.get('file_name')
        if fname:
            p = f"{val_dataset.root}/images/val2014/{fname}"
        else:
            p = None
        paths.append(p)
    return paths


def classify_image_by_texts(image_index_in_list, class_texts, caption_ids, text_matrix, val_dataset, image_ids, image_embs):
    """
    Given an image index (index into image_ids/image_embs), and a list of class text tokens,
    return similarity scores for each class and sorted classes.
    Classes are represented by averaging embeddings of captions that contain the class text.
    """
    ann_id_to_caption = val_dataset.annotations.get('ann_id_to_caption', {})
    cid_to_index = {cid: i for i, cid in enumerate(caption_ids)}

    class_embs = []
    class_names = []
    for cls in class_texts:
        matches = [cid for cid, txt in ann_id_to_caption.items() if cls.lower() in txt.lower()]
        idxs = [cid_to_index[int(m)] for m in matches if int(m) in cid_to_index]
        if not idxs:
            # use zeros if no matches
            emb = torch.zeros(text_matrix.size(1), device=text_matrix.device)
        else:
            emb = text_matrix[idxs].mean(dim=0)
        class_embs.append(emb)
        class_names.append(cls)

    class_embs = F.normalize(torch.stack(class_embs), dim=1)  # (C, D)
    img_emb = image_embs[image_index_in_list].to(DEVICE).unsqueeze(0)  # (1, D)
    sims = (img_emb @ class_embs.T).squeeze(0)  # (C,)
    scores = sims.tolist()
    ranked = sorted(zip(class_names, scores), key=lambda x: x[1], reverse=True)
    return ranked


def main():
    print("Loading dataset...")
    dataset = Coco2014Dataset(
        train_encodings_path="coco2014/train_caption_encodings.pt",
        val_encodings_path="coco2014/val_caption_encodings.pt",
        batch_size=BATCH_SIZE,
        n_workers=8,
        pin=True
    )

    dataset.setup()
    val_dataset = dataset.val_ds
    val_dataset._init_self()

    val_loader = dataset.val_dataloader()

    print("Loading model…")
    model = CLIPModel(
        projection_dim=512,
        freeze_backbone=False
    ).to(DEVICE)

    model.load_state_dict(torch.load("logs/initial_training/best_model.pth", map_location=DEVICE))
    model.eval()

    print("Building text embedding matrix…")
    caption_ids, text_matrix, caption_to_image = compute_text_matrix(val_dataset)

    print("Embedding images…")
    image_embs, true_caption_ids, image_ids = compute_image_embeddings(model, val_loader, val_dataset)
    image_embs = image_embs.to(DEVICE)

    print("Calculating Recall…")
    for k in [1, 5, 10]:
        i2t = recall_image_to_text(image_embs, text_matrix, true_caption_ids, caption_ids, k)
        t2i = recall_text_to_image(image_embs, text_matrix, caption_ids, caption_to_image, image_ids, k)
        print(f"K={k} | I2T Recall@{k}: {i2t*100:.2f}% | T2I Recall@{k}: {t2i*100:.2f}%")

    query = "sport"
    img_paths = retrieve_images_for_text_query(query, caption_ids, text_matrix, val_dataset, image_embs, image_ids, topk=5)
    if img_paths:
        out = os.path.join(OUT_DIR, f"text_query_{query}_top5.jpg")
        save_image_grid([p for p in img_paths if p is not None], out)
        print(f"Saved top-5 images for query '{query}' to {out}")

    class_texts = ["a person", "an animal", "a landscape"]
    ranked = classify_image_by_texts(0, class_texts, caption_ids, text_matrix, val_dataset, image_ids, image_embs)
    print("Class scores for image 0:", ranked)


if __name__ == "__main__":
    main()
