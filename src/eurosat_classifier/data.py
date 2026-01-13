from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(
    data_dir: str = "data/raw/eurosat/EuroSAT",
    batch_size: int = 64,
    valid_fraction: float = 0.2,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for EuroSAT RGB.


    """

    data_path = Path(data_dir)

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

    full_dataset = datasets.ImageFolder(root=str(data_path), transform=train_transform)

    # Train / validation split
    valid_size = int(len(full_dataset) * valid_fraction)
    train_size = len(full_dataset) - valid_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    valid_dataset.dataset.transform = valid_transform

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    validloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return trainloader, validloader