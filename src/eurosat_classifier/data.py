from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
from typing import cast

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset


@dataclass
class DataConfig:
    """
    Configuration container for the EuroSAT dataloaders.

    This keeps all data related hyperparameters in one place and makes it
    easy to pass them around or serialize them in configs.
    """

    data_dir: str = "data/raw/eurosat/EuroSAT"
    batch_size: int = 64
    valid_fraction: float = 0.2
    num_workers: int = 4


def get_dataloaders(config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for EuroSAT RGB.

    Args:
        config (DataConfig): Data configuration with paths and hyperparameters.

    Returns:
        Tuple[DataLoader, DataLoader]:
            - train dataloader
            - validation dataloader
    """
    data_path = Path(config.data_dir)

    # Transformations for the training split
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

    # Transformations for the validation split (deterministic, no augmentation)
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

    # Load the full dataset using ImageFolder
    full_dataset = datasets.ImageFolder(root=str(data_path), transform=train_transform)

    # Compute sizes for train and validation splits
    valid_size = int(len(full_dataset) * config.valid_fraction)
    train_size = len(full_dataset) - valid_size

    # Randomly split the dataset into train and validation subsets
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    # Use the validation transform on the validation subset

    # Tell mypy that the underlying dataset supports `.transform`
    base_valid = cast(VisionDataset, valid_dataset.dataset)
    base_valid.transform = valid_transform

    # Build the training dataloader
    trainloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Build the validation dataloader
    validloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return trainloader, validloader
