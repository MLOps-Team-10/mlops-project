from __future__ import annotations
import os

from pathlib import Path
from typing import Tuple

import time

import hydra
import torch
from hydra.utils import get_original_cwd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader

from eurosat_classifier.data import get_dataloaders, DataConfig
from eurosat_classifier.model import EuroSATModel, ModelConfig
from eurosat_classifier.scripts.download_data import ensure_eurosat_rgb
import wandb
from dotenv import load_dotenv


def setup_logging(logs_dir: Path) -> None:
    """
    Configure loguru for both console and file logging.

    We explicitly write logs under repo_root/logs so that runs are reproducible and
    logs don't end up inside Hydra's outputs/ working directory.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Remove default handler to fully control formatting, levels, and sinks
    logger.remove()

    # Console sink: readable progress at INFO level
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        level="INFO",
    )

    # File sink: detailed traceability at DEBUG level, with rotation/retention
    logger.add(
        logs_dir / "training.log",
        rotation="10 MB",
        retention="10 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
    )


def select_device() -> torch.device:
    """
    Select the best available device.

    Preference order:
    1) Apple Silicon GPU (MPS)
    2) NVIDIA GPU (CUDA)
    3) CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Validation loop executed after each epoch.

    Returns:
        (valid_loss, valid_accuracy) computed over the full validation set.

    Notes:
        - model.eval() disables training-only behavior (dropout, BN updates)
        - torch.no_grad() reduces memory usage and speeds up evaluation
    """
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0

    logger.debug("Validation started")

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            valid_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    logger.debug("Validation finished")

    # Dataset-average loss and accuracy
    return valid_loss / total, correct / total


def train(
    data_dir: str,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    num_workers: int,
    model_name: str,
    log_interval: int,
    valid_fraction: float,
    num_classes: int,
    pretrained: bool,
    logs_dir: Path,
    models_dir: Path,
) -> None:
    """
    Train a EuroSAT image classifier.

    MLOps notes:
    - Inputs are explicit args to ease CLI integration and experiment tracking.
    - Logs go to both console (INFO) and file (DEBUG) for auditability.
    - Checkpointing stores minimal metadata needed for later reload.
    """

    logger.info("=" * 80)
    logger.info("INITIALIZING TRAINING")
    logger.info("=" * 80)

    # Initialize wandb if not already done (for standalone train() calls, e.g., in tests)
    if not wandb.run:
        wandb.init(mode="disabled")

    device = select_device()
    logger.info(f"Device selected: {device}")

    # Data loading
    logger.info("Loading dataset and creating dataloaders...")
    data_config = DataConfig(
        data_dir=data_dir,
        batch_size=batch_size,
        valid_fraction=valid_fraction,
        num_workers=num_workers,
    )
    trainloader, validloader = get_dataloaders(config=data_config)

    # trainloader, validloader = get_dataloaders(
    # data_dir=data_dir,
    # batch_size=batch_size,
    # valid_fraction=valid_fraction,
    # num_workers=num_workers,
    # )

    logger.info(f"Training samples: {len(trainloader.dataset)}")
    logger.info(f"Validation samples: {len(validloader.dataset)}")

    # Model creation
    logger.info("Building model...")
    config = ModelConfig(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    model = EuroSATModel(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_name}")
    logger.info(f"Number of parameters: {num_params / 1_000_000:.2f}M")

    # Optimization setup
    criterion = nn.CrossEntropyLoss()  # expects raw logits
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info("Using Adam optimizer")
    logger.info(f"Learning rate: {learning_rate}")

    # Where checkpoints live (always repo_root/models)
    models_dir.mkdir(parents=True, exist_ok=True)

    best_valid_acc = 0.0

    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    for epoch in range(epochs):
        logger.info("")
        logger.info(f"===== Epoch {epoch + 1}/{epochs} =====")
        epoch_start = time.time()

        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (images, labels) in enumerate(trainloader):
            logger.info(f"batch idx: {batch_idx}")
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            batch_acc = (preds == labels).float().mean().item()
            wandb.log({"train/batch_loss": loss.item(), "train/batch_accuracy": batch_acc})

            running_loss += loss.item() * images.size(0)
            logger.info(f"Running loss: {running_loss}")

            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            if batch_idx % log_interval == 0:
                logger.debug(f"[Epoch {epoch + 1}] Batch {batch_idx}/{len(trainloader)} | Loss: {loss.item():.4f}")

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        logger.info(
            f"Epoch completed in {time.time() - epoch_start:.1f}s | "
            f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}"
        )

        logger.info("Running validation...")
        valid_loss, valid_acc = validate(model, validloader, criterion, device)

        logger.info(f"Validation loss: {valid_loss:.4f}")
        logger.info(f"Validation accuracy: {valid_acc:.4f}")

        # Save best checkpoint (by validation accuracy)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            ckpt_path = models_dir / "eurosat_best.pth"

            torch.save(
                {
                    "model_name": config.model_name,
                    "num_classes": config.num_classes,
                    "state_dict": model.state_dict(),
                },
                ckpt_path,
            )

            logger.info(f"Saved new best model (acc={best_valid_acc:.4f}) -> {ckpt_path}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best validation accuracy achieved: {best_valid_acc:.4f}")
    logger.info("=" * 80)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra entrypoint.

    Important:
    - Hydra changes current working directory to outputs/... by default.
    - We anchor all important paths to repo_root (original cwd).
    """
    repo_root = Path(get_original_cwd())
    load_dotenv(repo_root / ".env")
    if os.getenv("WANDB_API_KEY"):
        logger.info(" W&B API Key found in environment")
    else:
        logger.warning("W&B API Key NOT found!")

    logs_dir = repo_root / "logs"
    models_dir = repo_root / "models"

    # Resolve dataset directory relative to repo root for reproducibility
    data_dir = (repo_root / cfg.data.data_dir).resolve()

    # Configure logging before emitting any log lines
    setup_logging(logs_dir)
    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    logger.info("Hydra config:\n" + OmegaConf.to_yaml(cfg))
    logger.info(f"Resolved data_dir: {data_dir}")

    # Download/copy dataset only if missing (idempotent)
    ensure_eurosat_rgb(download_root=str(repo_root / "data" / "raw"))
    # guarantee config and bootstrap match
    expected = (repo_root / "data" / "raw" / "eurosat_rgb").resolve()
    if data_dir != expected:
        raise ValueError(f"Hydra data.data_dir={data_dir} does not match expected EuroSAT RGB dir {expected}")
    train(
        data_dir=str(data_dir),  # absolute resolved path
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        valid_fraction=cfg.data.valid_fraction,
        learning_rate=cfg.training.learning_rate,
        epochs=cfg.training.epochs,
        log_interval=cfg.training.log_interval,
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        logs_dir=logs_dir,
        models_dir=models_dir,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
