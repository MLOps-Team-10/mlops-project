from pathlib import Path
import argparse

import torch
from loguru import logger
from torch import nn

from eurosat_classifier.data import get_dataloaders, DataConfig
from eurosat_classifier.model import EuroSATModel, ModelConfig
from eurosat_classifier.train import validate


def select_device() -> torch.device:
    """Select the best available device."""
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA")
        return torch.device("cuda")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")


def load_model(ckpt_path: Path, device: torch.device) -> EuroSATModel:
    """Load a trained model from checkpoint."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    logger.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    config = ModelConfig(
        model_name=checkpoint["model_name"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
    )

    model = EuroSATModel(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate EuroSAT model")

    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (0 recommended in Docker)",
    )
    parser.add_argument(
        "--valid-fraction",
        type=int,
        default=0.2,
        help="Valid fraction",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()

    logger.info("Creating validation dataloader...")
    _, validloader = get_dataloaders(
        DataConfig(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            valid_fraction=args.valid_fraction,
            num_workers=args.num_workers,
        )
    )

    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint["state_dict"]

    # remove the prefix added by torch.compile / wrappers
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model = EuroSATModel(
        ModelConfig(
            model_name=checkpoint["model_name"],
            num_classes=checkpoint["num_classes"],
            pretrained=False,
        )
    ).to(device)

    # now load
    model.load_state_dict(new_state_dict)
    criterion = nn.CrossEntropyLoss()

    logger.info("Running evaluation...")
    val_loss, val_acc = validate(model, validloader, criterion, device)

    logger.info("=" * 80)
    logger.info(f"Validation loss: {val_loss:.4f}")
    logger.info(f"Validation accuracy: {val_acc:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
