#!/usr/bin/env python3
from __future__ import annotations
import os
import sys

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
from eurosat_classifier.scripts.download_data import ensure_eurosat_rgb_cloud
import wandb
from dotenv import load_dotenv

# Add src to sys.path to ensure imports work if not installed
# This is a fallback, though uv run usually handles it if installed
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def setup_logging(logs_dir: Path) -> None:
    """
    Configure loguru for both console and file logging.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
        level="INFO",
    )
    logger.add(
        logs_dir / "training_profiled.log",
        rotation="10 MB",
        retention="10 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
    )


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_models_path(repo_root: Path) -> Path:
    models_dir_raw = os.getenv("AIP_MODEL_DIR")
    gcs_path = Path("/gcs/dtu-mlops-eurosat/eurosat/models/")
    if models_dir_raw:
        if models_dir_raw.startswith("gs://"):
            raise ValueError(f"AIP_MODEL_DIR looks like a GCS URI ({models_dir_raw}).")
        return Path(models_dir_raw)

    if gcs_path.exists():
        return gcs_path

    return repo_root / "models"


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0

    logger.debug("Validation started")

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            # LIMIT FOR PROFILING
            if total >= 1000:
                logger.debug("Reached validation limit for profiling.")
                break

            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            valid_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0, 0.0

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
    logger.info("=" * 80)
    logger.info("INITIALIZING PROFILING TRAINING (SHORT)")
    logger.info("=" * 80)

    if not wandb.run:
        wandb.init(mode="disabled")

    device = select_device()
    logger.info(f"Device selected: {device}")

    data_config = DataConfig(
        data_dir=data_dir,
        batch_size=batch_size,
        valid_fraction=valid_fraction,
        num_workers=num_workers,
    )

    trainloader, validloader = get_dataloaders(config=data_config)

    config = ModelConfig(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    model = EuroSATModel(config).to(device)
    model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Force 1 epoch for profiling if not specified, but respect config
    # We will break early anyway

    for epoch in range(epochs):
        logger.info(f"===== Epoch {epoch + 1}/{epochs} =====")
        epoch_start = time.time()

        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (images, labels) in enumerate(trainloader):
            # LIMIT FOR PROFILING
            if running_total >= 1000:
                logger.info("Reached ~1000 images limit for profiling. Stopping epoch.")
                break

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            if batch_idx % log_interval == 0:
                logger.debug(f"[Epoch {epoch + 1}] Batch {batch_idx} | Loss: {loss.item():.4f}")

        if running_total == 0:
            logger.warning("No data processed!")
            break

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

        # We only need one epoch for profiling usually
        logger.info("Stopping after 1 epoch for profiling.")
        break


@hydra.main(version_base=None, config_path="../src/eurosat_classifier/conf", config_name="config")
def main(cfg: DictConfig) -> None:
    repo_root = Path(get_original_cwd())
    load_dotenv(repo_root / ".env")

    logs_dir = repo_root / "logs"
    models_dir = get_models_path(repo_root)
    data_dir = (repo_root / cfg.data.data_dir).resolve()

    setup_logging(logs_dir)
    wandb.init(mode="disabled")  # Force disabled for profiling to avoid noise

    logger.info("Hydra config:\n" + OmegaConf.to_yaml(cfg))

    ensure_eurosat_rgb_cloud(download_root=str(repo_root / "data" / "raw"))

    train(
        data_dir=str(data_dir),
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
