from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from eurosat_classifier.data import DataConfig, get_dataloaders


def test_get_dataloaders_with_real_data():
    """
    Test get_dataloaders using the real EuroSAT dataset on disk.

    This test assumes that the dataset is available at the path specified
    in DataConfig.data_dir. If the dataset is missing, the test is skipped.
    """

    # Use the default configuration (points to the real dataset)
    cfg = DataConfig()

    cfg.num_workers = 0  # For test stability across environments
    cfg.batch_size = 16  # Smaller batch size for testing

    data_path = Path(cfg.data_dir)

    # If the dataset is not present, skip the test instead of failing
    if not data_path.exists():
        pytest.skip(f"EuroSAT dataset not found at {data_path}")

    # Create the dataloaders from the real dataset
    trainloader, validloader = get_dataloaders(cfg)

    # Both outputs should be valid DataLoader instances
    assert isinstance(trainloader, DataLoader)
    assert isinstance(validloader, DataLoader)

    # The underlying datasets should not be empty
    assert len(trainloader.dataset) > 0
    assert len(validloader.dataset) > 0

    # Fetch one batch from the training loader
    x_train, y_train = next(iter(trainloader))

    # Basic checks on the training batch
    assert isinstance(x_train, torch.Tensor)
    assert isinstance(y_train, torch.Tensor)

    # Expected shape: (batch_size, C, H, W)
    assert x_train.ndim == 4

    # RGB images, so C should be 3
    assert x_train.shape[1] == 3
