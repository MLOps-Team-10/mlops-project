from __future__ import annotations

from pathlib import Path
from typing import Tuple, cast, Sized

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


import torch.profiler
from torch.profiler import ProfilerActivity


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
    profiler_logs_dir: Path,
) -> None:
    """
    Train a EuroSAT image classifier with PyTorch profiler enabled.
    """

    logger.info("=" * 80)
    logger.info("INITIALIZING PROFILING TRAINING")
    logger.info("=" * 80)

    device = select_device()
    logger.info(f"Device selected: {device}")

    # Data loading
    trainloader, validloader = get_dataloaders(
        DataConfig(
            data_dir=data_dir,
            batch_size=batch_size,
            valid_fraction=valid_fraction,
            num_workers=num_workers,
        )
    )

    # Model creation
    config = ModelConfig(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    model = EuroSATModel(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("STARTING PROFILING...")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    elif device.type == "mps":
        # Note: MPS profiling is experimental and might not provide full details
        pass

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiler_logs_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for epoch in range(epochs):
            logger.info(f"===== Epoch {epoch + 1}/{epochs} =====")
            model.train()
            for batch_idx, (images, labels) in enumerate(trainloader):
                # For profiling, we only need a few batches to get representative data
                if epoch == 0 and batch_idx >= 5:
                    logger.info("Reached batch limit for profiling. Stopping.")
                    break

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                # Signal the profiler that a step is complete
                prof.step()

            if epoch == 0 and any(batch_idx >= 5 for batch_idx, _ in enumerate(trainloader)):
                break

    logger.info("=" * 80)
    logger.info("PROFILING COMPLETE")
    logger.info(f"Profiler traces saved to: {profiler_logs_dir}")
    logger.info("=" * 80)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Hydra entrypoint for profiling.
    """
    repo_root = Path(get_original_cwd())
    logs_dir = repo_root / "logs"
    models_dir = repo_root / "models"
    profiler_logs_dir = repo_root / "logs" / "profiler"
    profiler_logs_dir.mkdir(exist_ok=True)

    data_dir = (repo_root / cfg.data.data_dir).resolve()
    setup_logging(logs_dir)

    logger.info("Hydra config for profiling:\n" + OmegaConf.to_yaml(cfg))
    ensure_eurosat_rgb(download_root=str(repo_root / "data" / "raw"))

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
        profiler_logs_dir=profiler_logs_dir,
    )


if __name__ == "__main__":
    main()
