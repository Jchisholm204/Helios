from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from typing import Optional
from dataset import Coco2014


class Coco2014Dataset():
    def __init__(self,
                 root='./coco2014',
                 image_size=(224, 224),
                 # Separate paths for train/val encodings for flexibility
                 train_encodings_path: Optional[str] = None,
                 val_encodings_path: Optional[str] = None,
                 # Separate transforms for augmentation during training
                 train_transform: Optional[transforms.Compose] = None,
                 val_transform: Optional[transforms.Compose] = None,
                 n_workers: int = 16,
                 batch_size: int = 128,
                 pin: bool = True):

        # Store configuration parameters
        self.root = root
        self.image_size = image_size
        self.train_encodings_path = train_encodings_path
        self.val_encodings_path = val_encodings_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.pin = pin

        # Dataset storage
        self.train_ds = None
        self.val_ds = None

    # --- Dataset Setup ---

    def setup(self):
        """Initializes the heavy-weight Dataset objects (triggers ONLY the __init__ of Coco2014)."""
        print("Initializing Datasets...")

        # Instantiate Training Dataset
        self.train_ds = Coco2014(
            root=self.root,
            image_size=self.image_size,
            encodings_path=self.train_encodings_path,
            is_train=True,
            transform=self.train_transform
        )

        # Instantiate Validation Dataset
        # Validation typically does not use heavy augmentation, so we use a separate transform
        self.val_ds = Coco2014(
            root=self.root,
            image_size=self.image_size,
            encodings_path=self.val_encodings_path,
            is_train=False,
            transform=self.val_transform
        )
        print("Dataset objects created. Data loading will be deferred until first access.")

    # --- DataLoader Accessors ---

    def train_dataloader(self):
        """Returns the DataLoader for the training split."""
        if self.train_ds is None:
            self.setup()  # Ensure datasets are initialized if not already

        # The DataLoader wraps the EXISTING dataset object
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,  # Crucial for training: shuffle the data order
            num_workers=self.n_workers,
            pin_memory=self.pin
        )

    def val_dataloader(self):
        """Returns the DataLoader for the validation split."""
        if self.val_ds is None:
            self.setup()  # Ensure datasets are initialized if not already
        
        ind = range(1000)
        limited = Subset(self.val_ds, ind)

        # The DataLoader wraps the EXISTING dataset object
        return DataLoader(
            limited,
            batch_size=self.batch_size,
            shuffle=False,  # Crucial for validation: do not shuffle the data order
            num_workers=self.n_workers,
            pin_memory=self.pin
        )
