import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import json
from typing import Dict, List, Any
import random


class Coco2014(Dataset):
    def __init__(self, root='./coco2014', image_size=(224, 224),
                 encodings_path: str = None, is_train=True,
                 transform: transforms.Compose = None):
        # Class setup
        self.root = root
        self.image_size = image_size
        self.is_train = is_train
        self.encodings_path = encodings_path
        self.is_init = False

        # Normalization Values (CLIP)
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)

        self.annotations: Dict[str, Any] = None
        self.image_ids: List[int] = None
        self.encodings: Dict[str, torch.Tensor] = None
        self.transform = transform

    def _init_self(self):
        if self.is_init:
            return
        # Load the json Labels file
        self.annotations = self._load_coco_annotations(self.is_train)
        self.image_ids = list(self.annotations['images'].keys())

        # Load the encoded labels if they exist
        if self.encodings_path is not None:
            print("Using Embedded Labels")
            self.encodings = self._load_encodings(self.encodings_path)
        else:
            self.enclabels = None

        # Create the transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            post_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
            self.transform = transforms.Compose(
                self.transform.transforms + post_transform.transforms)
        self.is_init = True

    def _load_encodings(self, fname: str):
        return torch.load(f'{fname}')

    def _load_coco_annotations(self, is_train: bool):
        fname = "captions_train2014.json" if is_train else "captions_val2014.json"
        with open(f"{self.root}/annotations/{fname}") as f:
            coco_data = json.load(f)

        image_id_to_meta = {img['id']: img for img in coco_data['images']}
        image_id_to_captions = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            caption = ann['caption'].strip()
            if img_id not in image_id_to_captions:
                image_id_to_captions[img_id] = []
            image_id_to_captions[img_id].append(caption)

        return {
            'images': image_id_to_meta,
            'annotations': image_id_to_captions
        }

    def __len__(self):
        self._init_self()
        return len(self.image_ids)

    def __getitem__(self, idx):
        self._init_self()
        img_id = self.image_ids[idx]
        img_meta = self.annotations['images'][img_id]

        # Pick a caption
        # captions = self.annotations['annotations'].get(img_id, ["no caption"])
        # caption = random.choice(captions)

        # Use precomputed embedding if available
        # if self.encodings is not None:
        #     if img_id not in self.encodings:
        #         raise KeyError(f"No embedding found for image_id {img_id}")
        #     label_output = random.choice(self.encodings[img_id])
        # else:
        #     label_output = caption

        # Load image
        img_filename = img_meta['file_name']
        img_path = f"{self.root}/images/train2014/{
            img_filename}" if self.is_train else f"{self.root}/images/val2014/{img_filename}"
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {
                  e}. Retrying with random image.")
            return self.__getitem__(random.randint(0, len(self)-1))

        return image, img_id
