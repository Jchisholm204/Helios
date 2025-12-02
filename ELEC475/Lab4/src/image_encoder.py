import os
import json
import time
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms as T

# Use your local model implementation
from core.model import CLIPModel

# --- Configuration Constants ---
OUTPUT_DIR = './coco2014/'
OUTPUT_FILENAME = 'val_image_encodings.pt'
COCO_ANNOTATIONS_FILE = 'coco2014/annotations/captions_val2014.json'
COCO_IMAGES_DIR = 'coco2014/images/val2014'
MODEL_WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "logs", "augmentation_20251130_105023", "best_model.pth")
BATCH_SIZE = 64
IMAGE_SIZE = 224

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class ImageEncoder:
    def __init__(self, weights_path: str = MODEL_WEIGHTS):
        self.weights_path = weights_path
        self.model = None
        # Use the same normalization as dataset/CLIP
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)
        self.transform = T.Compose([
            T.Resize(int(IMAGE_SIZE * 256 / 224)),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])

    def load_annotations(self) -> List[int]:
        if not os.path.exists(COCO_ANNOTATIONS_FILE):
            raise FileNotFoundError(f"Annotations file not found: {COCO_ANNOTATIONS_FILE}")
        with open(COCO_ANNOTATIONS_FILE, "r") as f:
            coco = json.load(f)
        img_ids = sorted({ann["image_id"] for ann in coco.get("annotations", [])})
        return img_ids

    def load_encoder(self):
        if self.model is not None:
            return
        # Instantiate your model (defaults match training code)
        model = CLIPModel().to(device)
        # Robust state dict loading (accept raw state_dict or dict with 'state_dict'/'model_state_dict')
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")
        sd = torch.load(self.weights_path, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            # fallback: try to find prefixed keys (e.g., 'module.' from DataParallel)
            new_sd = {}
            for k, v in sd.items():
                nk = k.replace("module.", "")
                new_sd[nk] = v
            model.load_state_dict(new_sd, strict=False)
        model.eval()
        self.model = model

    def load_image(self, img_id: int) -> Image.Image:
        filename = f"COCO_val2014_{img_id:012d}.jpg"
        path = os.path.join(COCO_IMAGES_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        img = Image.open(path).convert("RGB")
        return img

    def encode_images(self, image_ids: List[int]) -> Dict[int, List[torch.Tensor]]:
        if self.model is None:
            raise RuntimeError("Encoder not loaded; call load_encoder() first.")
        all_embeddings: Dict[int, List[torch.Tensor]] = {}
        start = time.time()
        imgs_batch = []
        ids_batch = []
        for iid in image_ids:
            try:
                img = self.load_image(iid)
            except FileNotFoundError:
                # skip missing images
                continue
            imgs_batch.append(self.transform(img))
            ids_batch.append(iid)
            if len(imgs_batch) >= BATCH_SIZE:
                batch = torch.stack(imgs_batch).to(device)
                with torch.no_grad():
                    vecs = self.model(batch)  # model returns normalized projected embeddings
                    vecs = vecs.cpu()
                for idx, img_id in enumerate(ids_batch):
                    all_embeddings[img_id] = [vecs[idx]]
                imgs_batch = []
                ids_batch = []
        # remaining
        if imgs_batch:
            batch = torch.stack(imgs_batch).to(device)
            with torch.no_grad():
                vecs = self.model(batch)
                vecs = vecs.cpu()
            for idx, img_id in enumerate(ids_batch):
                all_embeddings[img_id] = [vecs[idx]]
        end = time.time()
        print(f"Encoded {len(all_embeddings)} images in {end - start:.2f}s")
        return all_embeddings


if __name__ == "__main__":
    enc = ImageEncoder(MODEL_WEIGHTS)
    try:
        image_ids = enc.load_annotations()
    except FileNotFoundError as e:
        print(f"Fatal: {e}")
        raise SystemExit(1)

    enc.load_encoder()
    embeddings = enc.encode_images(image_ids)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    torch.save(embeddings, save_path)
    print(f"Saved image encodings to {save_path}")