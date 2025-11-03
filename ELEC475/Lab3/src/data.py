import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import numpy as np
import random


class PascalVOCSegmentation:
    def __init__(self, root='./data', image_size=(256, 256),
                 batch_size=8, n_workers=8, download=False):
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.n_workers = n_workers

        # Normalization for pretrained encoders
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.download = download

    # ---------------------
    # Joint augmentation for image + mask
    # ---------------------
    def joint_transform(self, image, mask):
        # Random horizontal flip
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # Random resized crop with shared params
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.5, 2.0), ratio=(1.0, 1.0)
        )
        image = F.resized_crop(image, i, j, h, w, size=self.image_size)
        mask = F.resized_crop(mask, i, j, h, w, size=self.image_size,
                              interpolation=F.InterpolationMode.NEAREST)
        return image, mask

    # ---------------------
    # Dataset subclass with joint transform
    # ---------------------
    class _VOCSegmentationJoint(datasets.VOCSegmentation):
        def __init__(self, outer, *args, train=True, **kwargs):
            super().__init__(*args, **kwargs)
            self.outer = outer
            self.train = train

        def __getitem__(self, index):
            image, mask = super().__getitem__(index)

            # Apply joint transform only during training
            if self.train:
                image, mask = self.outer.joint_transform(image, mask)
            else:
                image = F.resize(image, self.outer.image_size)
                mask = F.resize(mask, self.outer.image_size,
                                interpolation=F.InterpolationMode.NEAREST)

            # Convert to tensor + normalize
            image = F.to_tensor(image)
            image = F.normalize(
                image, mean=self.outer.mean, std=self.outer.std)

            # Convert mask to tensor (long type)
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
            return image, mask

    # ---------------------
    # DataLoader getters
    # ---------------------
    def get_train_loader(self):
        train_dataset = self._VOCSegmentationJoint(
            outer=self,
            root=self.root,
            year='2012',
            image_set='train',
            download=self.download,
            train=True
        )
        return DataLoader(train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.n_workers,
                          pin_memory=False)

    def get_val_loader(self):
        val_dataset = self._VOCSegmentationJoint(
            outer=self,
            root=self.root,
            year='2012',
            image_set='val',
            download=self.download,
            train=False
        )
        return DataLoader(val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.n_workers,
                          pin_memory=False)


if __name__ == "__main__":
    voc_data = PascalVOCSegmentation(
        root="./data",
        image_size=(256, 256),
        batch_size=8,
        n_workers=8
    )

    train_loader = voc_data.get_train_loader()
    val_loader = voc_data.get_val_loader()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
