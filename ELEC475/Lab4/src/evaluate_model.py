import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.model import CLIPModel
from dataloader import Coco2014Dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128


@torch.no_grad()
def compute_text_matrix(val_dataset):
    """
    val_dataset.encodings is:
        { caption_id (int) : [embedding1, embedding2, ...] }
    Each embedding is a tensor of dimension D.
    """
    encs = val_dataset.encodings

    caption_ids = list(encs.keys())
    text_embs = []

    for cid in caption_ids:
        variants = encs[cid]                   # list of tensors
        stacked = torch.stack(variants)        # (num_variants, D)
        mean_emb = stacked.mean(dim=0)         # (D,)
        text_embs.append(mean_emb)

    text_embs = torch.stack(text_embs)         # (N_text, D)
    text_embs = F.normalize(text_embs, dim=1)

    return caption_ids, text_embs.to(DEVICE)


@torch.no_grad()
def compute_image_embeddings(model, val_loader):
    image_embs = []
    true_caption_ids = []

    for images, caption_id_emb_list in tqdm(val_loader, desc="Images"):

        # images: [B, 3, H, W]
        images = images.to(DEVICE)

        # caption_id_emb_list: list[list of embeddings], but the label is the key (caption_id)
        # Dataset must return the caption_id also.
        # Let's modify dataset to do that:
        # In your dataset, instead of returning (image, embeddings), also return caption_id.

        # For now, assume val_loader returns (images, caption_ids)
        caption_ids = caption_id_emb_list     # rename

        feats = model(images)
        feats = F.normalize(feats, dim=1)
        image_embs.append(feats.cpu())
        true_caption_ids.extend(caption_ids)

    return torch.cat(image_embs), true_caption_ids


def recall_at_k(image_emb, text_emb, true_caption_ids, caption_ids, k):
    """
    image_emb: (N, D)
    text_emb:  (M, D)
    true_caption_ids: length N list
    caption_ids: length M list
    """
    # similarity matrix (N, M)
    sims = image_emb @ text_emb.T

    # map caption ID → index in text matrix
    cid_to_index = {cid: i for i, cid in enumerate(caption_ids)}
    gt = torch.tensor([cid_to_index[c] for c in true_caption_ids])

    # top-k retrieval
    topk = sims.topk(k, dim=1).indices        # (N, k)

    correct = (topk == gt.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


def main():
    print("Loading dataset...")
    dataset = Coco2014Dataset(
        train_encodings_path="coco2014/train_caption_encodings.pt",
        val_encodings_path="coco2014/val_caption_encodings.pt",
        batch_size=BATCH_SIZE,
        n_workers=8,
        pin=True
    )

    val_dataset = dataset.val_ds
    val_dataset._init_self()

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8
    )

    print("Loading model…")
    model = CLIPModel(
        projection_dim=512,
        freeze_backbone=False
    ).to(DEVICE)

    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    model.eval()

    print("Building text embedding matrix…")
    caption_ids, text_matrix = compute_text_matrix(val_dataset)

    print("Embedding images…")
    image_embs, true_caption_ids = compute_image_embeddings(model, val_loader)
    image_embs = image_embs.to(DEVICE)

    print("Calculating Recall…")
    for k in [1, 5, 10]:
        r = recall_at_k(image_embs, text_matrix,
                        true_caption_ids, caption_ids, k)
        print(f"Recall@{k}: {r*100:.2f}%")


if __name__ == "__main__":
    main()
