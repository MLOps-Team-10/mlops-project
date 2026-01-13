from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class EuroSATDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for EuroSAT.

    Why a DataModule:
    - Keeps all data-loading logic (transforms, splits, dataloaders) in one place
    - Trainer can call setup() once and reuse the dataloaders across epochs
    - Makes your training script smaller and more reproducible

    This is a minimal refactor of your original get_dataloaders() function:
    - Same transforms
    - Same ImageFolder dataset
    - Same random_split train/val
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        valid_fraction: float,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.valid_fraction = valid_fraction
        self.num_workers = num_workers

        self.train_dataset = None
        self.valid_dataset = None

        # Transforms are defined once and reused in setup().
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
        Called by Lightning to prepare datasets.

        stage can be:
        - "fit": training/validation
        - "test": testing
        - None: setup everything (common during development)

        We only need train/val here.
        """
        data_path = Path(self.data_dir).expanduser().resolve()

        # Start with training transforms for the full dataset.
        full_dataset = datasets.ImageFolder(
            root=str(data_path),
            transform=self.train_transform,
        )

        # Compute split sizes exactly like your original code.
        valid_size = int(len(full_dataset) * self.valid_fraction)
        train_size = len(full_dataset) - valid_size

        self.train_dataset, self.valid_dataset = random_split(
            full_dataset, [train_size, valid_size]
        )

        # Important: change transform only for the validation subset.
        # random_split returns Subset objects; the underlying dataset is shared.
        # This matches your original approach.
        self.valid_dataset.dataset.transform = self.valid_transform

    def train_dataloader(self) -> DataLoader:
        """Training DataLoader used by Trainer.fit()."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            # Optional: can speed up on repeated epochs when num_workers > 0
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation DataLoader used by Trainer.fit()."""
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )