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
            self.encodings = self._load_encodings(self.is_train)
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

    def _load_encodings(self, is_train: bool):
        if is_train:
            fname = f"{self.root}/encodings/train.pt"
        else:
            fname = f"{self.root}/encodings/val.pt"
        return torch.load(f'{fname}')

    def _load_coco_annotations(self, is_train: str):
        if is_train:
            fname = "instances_train2014.json"
        else:
            fname = "instances_val2014.json"

        with open(f"{self.root}/annotations/{fname}") as f:
            coco_data = json.load(f)
            image_id_to_meta = {
                img['id']: img for img in coco_data['images']
            }
            image_id_to_annotations = {}
            for ann in coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in image_id_to_annotations:
                    image_id_to_annotations[img_id] = []
                image_id_to_annotations[img_id].append(ann)

            category_id_to_name = {
                cat['id']: cat['name'] for cat in coco_data['categories']
            }
            return {
                'images': image_id_to_meta,
                'annotations': image_id_to_annotations,
                'categories': category_id_to_name
            }

    def __len__(self):
        self._init_self()
        return len(self.image_ids)

    def __getitem__(self, idx):
        self._init_self()
        # Get image id and metadata
        img_id = self.image_ids[idx]
        img_meta = self.annotations['images'][img_id]

        # Determine text label
        anns = self.annotations['annotations'].get(img_id, [])
        label_text = "no object"
        if anns:
            first_cat_id = anns[0]['category_id']
            label_text = self.annotations['categories'][first_cat_id]

        if self.encodings is not None:
            label_output = self.encodings.get(label_text, None)
            if label_output is None:
                raise KeyError(f"Missing pre-encoded label for: {label_text}")
        else:
            label_output = label_text

        img_filename = img_meta['file_name']
        if self.is_train:
            img_path = f"{self.root}/images/train2014/{img_filename}"
        else:
            img_path = f"{self.root}/images/val2014/{img_filename}"
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Retrying..")
            return self.__getitem__(random.randint(0, len(self)-1))

        return image, label_output
