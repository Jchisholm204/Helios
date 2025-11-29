import torch
import os
from typing import Dict, List
from transformers import CLIPTokenizer, CLIPTextModel
from torch.nn.functional import cosine_similarity
import time
import json

# --- Configuration Constants ---
OUTPUT_DIR = './coco2014/'
OUTPUT_FILENAME = 'val_caption_encodings.pt'
COCO_ANNOTATIONS_FILE = 'coco2014/annotations/captions_val2014.json'
ENCODER_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class CaptionEncoder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_annotations(self, filepath: str) -> Dict[int, List[str]]:
        """
        Returns a dictionary: image_id -> list of captions
        """
        print(f"Loading COCO captions from {filepath}...")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r') as f:
            coco_data = json.load(f)

        image_to_captions: Dict[int, List[str]] = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            caption = ann['caption'].strip()
            if img_id not in image_to_captions:
                image_to_captions[img_id] = []
            image_to_captions[img_id].append(caption)

        print(f"Loaded captions for {len(image_to_captions)} images.")
        return image_to_captions

    def load_encoder(self):
        if self.tokenizer is None:
            print(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)

        if self.model is None:
            print(f"Loading text model: {self.model_name}")
            self.model = CLIPTextModel.from_pretrained(
                self.model_name).to(device).eval()

        print("Encoder and Tokenizer loaded successfully.")

    def encode_captions(self, image_to_captions: Dict[int, List[str]]) -> Dict[int, List[torch.Tensor]]:
        """
        Returns a dict: image_id -> list of embedding tensors
        """
        if self.model is None:
            raise RuntimeError(
                "Encoder model not loaded. Call load_encoder() first.")

        all_image_embeddings: Dict[int, List[torch.Tensor]] = {}
        start_time = time.time()
        print("Encoding captions...")

        for img_id, captions in image_to_captions.items():
            # Prepare prompts
            texts = [f"a photo of {c}" for c in captions]

            # Tokenize batch
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Compute embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding_vectors = outputs.pooler_output.cpu()  # Move to CPU for storage

            all_image_embeddings[img_id] = [vec for vec in embedding_vectors]

        end_time = time.time()
        print(f"Encoding complete in {end_time - start_time:.2f} seconds.")
        return all_image_embeddings

    def verify_embeddings(self, embeddings: Dict[int, List[torch.Tensor]]):
        """
        Sanity check: pick two captions from the same image vs captions from different images
        """
        print("\n--- SANITY CHECK ---")
        img_ids = list(embeddings.keys())
        if len(img_ids) < 2:
            print("Not enough images for sanity check.")
            return

        # Pick first image (two captions)
        vec1, vec2 = embeddings[img_ids[0]][0], embeddings[img_ids[0]][-1]

        # Pick a caption from a different image
        vec3 = embeddings[img_ids[1]][0]

        sim_same = cosine_similarity(
            vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
        sim_diff = cosine_similarity(
            vec1.unsqueeze(0), vec3.unsqueeze(0)).item()

        print(f"Similarity(same image captions): {sim_same:.4f}")
        print(f"Similarity(different image captions): {sim_diff:.4f}")

        if sim_same > sim_diff:
            print(
                "SANITY CHECK PASS: Captions from the same image are closer than different images.")
        else:
            print("SANITY CHECK FAIL: Check model or tokenizer.")
        print("-------------------")


# --- Main Execution ---
if __name__ == '__main__':
    encoder = CaptionEncoder(ENCODER_MODEL_NAME)

    # Load COCO captions
    try:
        image_to_captions = encoder.load_annotations(COCO_ANNOTATIONS_FILE)
    except FileNotFoundError as e:
        print(f"Fatal Error: {e}")
        exit()

    # Load CLIP model
    encoder.load_encoder()

    # Encode all captions
    image_embeddings = encoder.encode_captions(image_to_captions)

    # Sanity check
    encoder.verify_embeddings(image_embeddings)

    # Save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    try:
        torch.save(image_embeddings, save_path)
        print(f"SUCCESS: Saved embeddings to {save_path}")
    except Exception as e:
        print(f"ERROR: Failed to save embeddings: {e}")
