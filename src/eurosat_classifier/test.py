# src/eurosat_classifier/test.py

from pathlib import Path

import torch
from loguru import logger

from eurosat_classifier.data import get_dataloaders
from eurosat_classifier.model import EuroSATModel, ModelConfig
from eurosat_classifier.train import validate  # reuse the validation logic from train.py


def select_device():
    """
    Select the best available computation device.

    Priority order:
    1) MPS for Apple Silicon (Metal backend)
    2) CUDA for NVIDIA GPUs
    3) CPU as a safe fallback

    Centralizing device selection avoids duplication and ensures
    consistent behavior across training and evaluation scripts.
    """
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA")
        return torch.device("cuda")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")


def main():
    """
    Entry point for offline model evaluation.

    This script:
    - rebuilds the validation dataloader with the same preprocessing as training
    - restores the best model checkpoint
    - evaluates the model on the validation split
    - reports final loss and accuracy

    Keeping evaluation separate from training is a best practice in MLOps,
    as it allows reproducible, isolated model assessment.
    """
    device = select_device()

    # Recreate dataloaders.
    # Using the same data pipeline as training is critical for fair evaluation.
    _, validloader = get_dataloaders(
        data_dir="data/raw/rgb",
        batch_size=64,
        valid_fraction=0.2,
        num_workers=4,
    )

    # Load the best checkpoint saved during training.
    ckpt_path = Path("models/eurosat_best.pth")
    logger.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Reconstruct model configuration from checkpoint metadata.
    # pretrained=False avoids reloading ImageNet weights, as we restore trained weights instead.
    config = ModelConfig(
        model_name=checkpoint["model_name"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
    )

    model = EuroSATModel(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    # Loss function used for evaluation.
    # Must match the one used during training for consistent metrics.
    criterion = torch.nn.CrossEntropyLoss()

    # Run evaluation using the shared validation function.
    val_loss, val_acc = validate(model, validloader, criterion, device)

    logger.info(f"Final validation loss: {val_loss:.4f}")
    logger.info(f"Final validation accuracy: {val_acc:.4f}")


# Allows the script to be executed directly without triggering evaluation on import.
if __name__ == "__main__":
    main()
