from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class EuroSATDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for the EuroSAT dataset.

    This class encapsulates all data-related logic:
    - dataset loading
    - train/validation splitting
    - image transformations
    - DataLoader construction

    Using a LightningDataModule allows the data pipeline
    to be reused across training, validation, testing,
    and deployment without duplicating code.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        valid_fraction: float,
        num_workers: int,
    ) -> None:
        """
        Args:
            data_dir: Path to the root directory of the EuroSAT dataset.
            batch_size: Number of samples per batch.
            valid_fraction: Fraction of the dataset reserved for validation.
            num_workers: Number of subprocesses used for data loading.
        """
        super().__init__()

        # Store configuration parameters
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.valid_fraction = valid_fraction
        self.num_workers = num_workers

        # Datasets are initialized in setup()
        self.train_dataset = None
        self.valid_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Create datasets and apply transformations.

        This method is called by the Lightning Trainer at the
        beginning of training/validation/testing.

        The 'stage' argument can be used to differentiate between
        'fit', 'validate', 'test', or 'predict' phases, but it is
        optional and not required in this use case.
        """

        # Training transformations:
        # - Resize to match ImageNet-sized backbones (e.g. ResNet)
        # - Random horizontal flip for data augmentation
        # - Normalize using ImageNet statistics
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Validation transformations:
        # - No data augmentation
        # - Same resizing and normalization as training
        valid_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Resolve dataset path to an absolute path
        data_path = Path(self.data_dir).expanduser().resolve()

        # Load the full dataset using ImageFolder
        # Class labels are inferred from subdirectory names
        full_dataset = datasets.ImageFolder(
            root=str(data_path),
            transform=train_transform,
        )

        # Compute train/validation split sizes
        valid_size = int(len(full_dataset) * self.valid_fraction)
        train_size = len(full_dataset) - valid_size

        # Split the dataset deterministically
        self.train_dataset, self.valid_dataset = random_split(
            full_dataset, [train_size, valid_size]
        )

        # Override the transform for the validation subset
        # (random_split keeps a reference to the same dataset object)
        self.valid_dataset.dataset.transform = valid_transform

    def train_dataloader(self) -> DataLoader:
        """
        Return the DataLoader used during training.

        Lightning will automatically call this method when
        running trainer.fit().
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return the DataLoader used during validation.

        Lightning will automatically call this method during
        validation loops.
        """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )