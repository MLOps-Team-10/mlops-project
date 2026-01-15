from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class EuroSATDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the EuroSAT dataset.

    Using a LightningDataModule (MLOps perspective):
    - Centralizes *all* data-related logic in a single, reusable component
    - Decouples data handling from model and training logic
    - Ensures consistent dataset splits and transforms across runs
    - Allows the Trainer to manage lifecycle hooks (setup, teardown see stages)

    This DataModule is a structured refactor of a classic PyTorch pipeline:
    - torchvision.datasets.ImageFolder
    - random train/validation split
    - separate transforms for train and validation
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        valid_fraction: float,
        num_workers: int,
    ) -> None:
        super().__init__()

        # Store configuration parameters.
        # These should come from Hydra to keep the pipeline fully config-driven.
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.valid_fraction = valid_fraction
        self.num_workers = num_workers

        # These will be populated in setup().
        self.train_dataset = None
        self.valid_dataset = None

        # Training-time transforms.
        # Includes data augmentation to improve generalization.
        self.train_transform = transforms.Compose(
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

        # Validation-time transforms.
        # No augmentation, only deterministic preprocessing.
        self.valid_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Prepare datasets for different stages of the training lifecycle.

        Lightning calls this method automatically:
        - stage="fit"  -> training + validation
        - stage="test" -> testing
        - stage=None   -> setup everything (common during development)

        Here we only implement train/validation logic.
        """
        # Resolve dataset path explicitly.
        # This avoids issues with relative paths when Hydra changes the working directory.
        data_path = Path(self.data_dir).expanduser().resolve()

        # Load the full dataset using ImageFolder.
        # At this point we apply training transforms by default.
        full_dataset = datasets.ImageFolder(
            root=str(data_path),
            transform=self.train_transform,
        )

        # Compute split sizes deterministically.
        # This mirrors a standard manual PyTorch split.
        valid_size = int(len(full_dataset) * self.valid_fraction)
        train_size = len(full_dataset) - valid_size

        # random_split returns Subset objects that share the same underlying dataset.
        self.train_dataset, self.valid_dataset = random_split(
            full_dataset,
            [train_size, valid_size],
        )

        # Important detail:
        # random_split does NOT clone the dataset.
        # We therefore override the transform only for the validation subset.
        self.valid_dataset.dataset.transform = self.valid_transform

    def train_dataloader(self) -> DataLoader:
        """
        DataLoader used during training.

        Key choices:
        - shuffle=True: required for SGD-based optimization
        - pin_memory=True: improves host-to-device transfer (especially with GPUs)
        - persistent_workers: avoids worker re-spawn at every epoch
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """
        DataLoader used during validation.

        No shuffling to ensure deterministic evaluation.
        """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
