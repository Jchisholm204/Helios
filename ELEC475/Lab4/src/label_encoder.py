import torch
import os
from typing import Dict, List
from transformers import CLIPTokenizer, CLIPTextModel
from torch.nn.functional import cosine_similarity
import time
import json

# --- Configuration Constants ---
OUTPUT_DIR = './coco2014/'
OUTPUT_FILENAME = 'label_encodings.pt'
# File containing categories, one per line
CATEGORY_FILE = 'coco2014/annotations/instances_train2014.json'
ENCODER_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512


class LabelEncoder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    # def load_categories(self, filepath: str) -> List[str]:
    #     print(f"Loading categories from: {filepath}")
    #     if not os.path.exists(filepath):
    #         raise FileNotFoundError(f"Category file not found at: {filepath}")
    #
    #     with open(filepath, 'r') as f:
    #         # Read lines, strip whitespace, and filter out empty strings
    #         categories = [line.strip() for line in f if line.strip()]
    #
    #     if "no object" not in categories:
    #         categories.append("no object")
    #
    #     print(f"Successfully loaded {len(categories)} categories.")
    #     return categories

    def load_categories(self, filepath: str) -> List[str]:
        print(f"Loading categories from official COCO JSON: {filepath}")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Annotation file not found at: {
                    filepath}. Please ensure CATEGORY_FILE points to the official COCO JSON (e.g., 'instances_train2014.json')."
            )

        try:
            with open(filepath, 'r') as f:
                coco_data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from file: {
                             filepath}. Check file integrity.")
        except Exception as e:
            raise IOError(f"Failed to read or process file: {e}")

        # Extract the names from the 'categories' list in the COCO JSON structure
        categories = []

        # Check if the 'categories' key exists
        category_list_from_json = coco_data.get('categories')
        if not category_list_from_json:
            raise KeyError(
                "JSON file is missing the required 'categories' array.")

        for cat_info in category_list_from_json:
            # COCO names are stored under the 'name' key; converting to lowercase for consistent dictionary keys
            if 'name' in cat_info:
                category_name = cat_info['name'].lower()
                categories.append(category_name)

        # 1. CRITICAL FIX: Add the required fallback label for unannotated images
        if "no object" not in categories:
            categories.append("no object")
            print(
                "INFO: Added 'no object' to the category list for encoding compatibility.")

        # Basic verification
        expected_size = 81
        if len(categories) != expected_size:
            print(f"WARNING: Category list size is {len(categories)}. Expected {
                  expected_size} (80 COCO + 'no object').")

        print(f"Successfully loaded {
              len(categories)} unique categories for encoding.")
        return categories

    def load_encoder(self):
        if self.tokenizer is None:
            print(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)

        if self.model is None:
            print(f"Loading text model: {self.model_name}")
            # Ensure model is set to evaluation mode
            self.model = CLIPTextModel.from_pretrained(self.model_name).eval()

        print("Encoder and Tokenizer loaded successfully.")

    def encode_labels(self, category_list: List[str]) -> Dict[str, torch.Tensor]:
        if self.model is None:
            raise RuntimeError(
                "Encoder model not loaded. Call load_encoder() first.")

        embeddings = {}

        # Prepare text inputs with a useful prompt template
        texts = [f"a photo of a {category}" for category in category_list]

        # Batch tokenize the entire list
        inputs = self.tokenizer(texts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True)

        print(f"Tokenizing {len(texts)} labels and generating embeddings...")
        start_time = time.time()

        # Run model inference without gradient tracking
        with torch.no_grad():
            output = self.model(**inputs)

            # Extract the pooled output (often used for classification/feature representation)
            # The shape will be [batch_size, EMBEDDING_DIM]
            embedding_vectors = output.pooler_output.cpu()

        end_time = time.time()
        print(f"Encoding complete in {end_time - start_time:.2f} seconds.")

        # Map category names back to their corresponding embedding tensors
        for category, vector in zip(category_list, embedding_vectors):
            embeddings[category] = vector

        return embeddings

    def verify_embeddings(self, embeddings: Dict[str, torch.Tensor]):
        """
        Performs a critical sanity check using cosine similarity on related/unrelated pairs.

        In a good feature space, related items (cat/dog) should have higher cosine
        similarity (closer to 1) than unrelated items (cat/chair).
        """

        print("\n--- EMBEDDING SANITY CHECK (Cosine Similarity) ---")

        # 1. Related Pair Check (e.g., animals)
        vec_cat = embeddings.get('cat')
        vec_dog = embeddings.get('dog')

        # 2. Unrelated Pair Check (e.g., animal vs furniture)
        vec_chair = embeddings.get('chair')

        if not all([vec_cat is not None, vec_dog is not None, vec_chair is not None]):
            print(
                "WARNING: One or more sanity check categories (cat, dog, chair) not found.")
            return

        # Cosine similarity requires the tensors to be 2D, so we unsqueeze them
        cat_dog_sim = cosine_similarity(
            vec_cat.unsqueeze(0), vec_dog.unsqueeze(0)).item()
        cat_chair_sim = cosine_similarity(
            vec_cat.unsqueeze(0), vec_chair.unsqueeze(0)).item()

        print(f"Similarity(cat, dog):   {cat_dog_sim:.4f}")
        print(f"Similarity(cat, chair): {cat_chair_sim:.4f}")

        # The key verification step
        if cat_dog_sim > cat_chair_sim:
            print(
                "VERIFICATION PASS: Related items (cat/dog) are closer than unrelated (cat/chair).")
        else:
            print(
                "VERIFICATION FAIL: Related items were found to be less similar. Check model loading.")
        print("-----------------------------------------------------")


# --- 2. Main Execution ---

if __name__ == '__main__':

    # 1. Initialize the Encoder Class
    encoder = LabelEncoder(ENCODER_MODEL_NAME)

    # 2. Load Categories from File
    try:
        categories = encoder.load_categories(CATEGORY_FILE)
    except FileNotFoundError as e:
        print(f"Fatal Error: {e}")
        exit()

    # 3. Load the Encoder Model
    encoder.load_encoder()

    # 4. Generate Embeddings
    label_embeddings = encoder.encode_labels(categories)

    encoder.verify_embeddings(label_embeddings)

    # 5. Create the output directory and save
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    try:
        torch.save(label_embeddings, save_path)
        print(f"\nSUCCESS: Label embeddings saved to {save_path}")
        print(f"Saved {len(label_embeddings)} embeddings, each of dimension {
              label_embeddings[categories[0]].shape[0]}.")

    except Exception as e:
        print(f"\nERROR: Failed to save embeddings: {e}")
