from pathlib import Path
import time

import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim
from loguru import logger

from eurosat_classifier.data import get_dataloaders
from eurosat_classifier.model import EuroSATModel, ModelConfig


def setup_logging() -> None:
    """Configure loguru for both console and file logging."""
    # Centralize logs in a dedicated folder to keep the project root clean
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Remove the default handler to fully control formatting/levels/sinks
    logger.remove()

    # Console logger:
    # - INFO level for readable training progress
    # - human-friendly timestamps and source locations for debugging
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level:<8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level="INFO",
    )

    # File logger:
    # - DEBUG level for detailed traceability (batch-level logs, validation phases, etc.)
    # - rotation/retention to avoid unbounded log growth on long runs
    logger.add(
        logs_dir / "training.log",
        rotation="10 MB",        # Rotate when file exceeds 10 MB
        retention="10 days",     # Keep logs for 10 days
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
               "{name}:{function}:{line} - {message}",
    )


def train(
    data_dir: str ,
    batch_size: int ,
    learning_rate: float ,
    epochs: int ,
    num_workers: int ,
    model_name: str ,
    log_interval: int ,
    valid_fraction: float ,
    num_classes: int ,
    pretrained: bool ,
) -> None:
    """
    Train a EuroSAT image classifier.

    - Hyperparameters are function arguments to ease CLI/experiment integration.
    - Logging is structured to support both human monitoring (console) and auditing (file).
    - The model returns logits (not probabilities) to work correctly with CrossEntropyLoss.
    """
    setup_logging()

    logger.info("=" * 80)
    logger.info("INITIALIZING TRAINING")
    logger.info("=" * 80)

    # Device selection:
    # - Prefer Apple Silicon GPU (MPS) when available
    # - Otherwise use CUDA if present, else fallback to CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device selected: {device}")

    # Load dataset and create dataloaders.
    # valid_fraction enforces a stable split at the dataloader level (depending on implementation).
    logger.info("Loading dataset and creating dataloaders...")


    trainloader, validloader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        valid_fraction=valid_fraction,
        num_workers=num_workers,
    )
    logger.info(f"Training samples: {len(trainloader.dataset)}")
    logger.info(f"Validation samples: {len(validloader.dataset)}")

    # Build the model.
    # timm handles backbone creation and (optionally) pretrained weights.
    logger.info("Building model...")
    config = ModelConfig(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    model = EuroSATModel(config).to(device)

    # Parameter count is a quick sanity check and useful metadata for experiment logs.
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_name}")
    logger.info(f"Number of parameters: {num_params / 1_000_000:.2f}M")

    # Loss and optimizer:
    # CrossEntropyLoss expects logits; do not apply softmax in the model.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info("Using Adam optimizer")
    logger.info(f"Learning rate: {learning_rate}")

    # Directory for saving model checkpoints.
    # Keeping checkpoints under /models makes it easy to add CI/CD artifacts later.
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Track the best validation accuracy for model selection.
    best_valid_acc = 0.0

    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    # Epoch-level training loop.
    # Each epoch:
    # 1) train on the training set
    # 2) evaluate on the validation set
    # 3) checkpoint if validation improves
    for epoch in range(epochs):
        logger.info("")
        logger.info(f"===== Epoch {epoch + 1}/{epochs} =====")
        epoch_start = time.time()

        model.train()  # Enable training mode (e.g. dropout, batchnorm updates)
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        # Iterate over training batches
        for batch_idx, (images, labels) in enumerate(trainloader):
            # Move data to the selected device
            images, labels = images.to(device), labels.to(device)

            # Standard training step:
            # - clear gradients
            # - forward pass
            # - compute loss
            # - backward pass
            # - optimizer step
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Update running metrics (accumulated across the epoch).
            # Multiply loss by batch size to compute a true dataset-average loss later.
            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            # Batch-level logs at DEBUG so console stays clean but file contains details.
            if batch_idx % log_interval == 0:
                logger.debug(
                    f"[Epoch {epoch+1}] Batch {batch_idx}/{len(trainloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        # Compute epoch metrics (dataset-average loss, accuracy).
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        logger.info(
            f"Epoch completed in {time.time() - epoch_start:.1f}s | "
            f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}"
        )

        # Validation step:
        # Run under no_grad + eval mode for correctness and performance.
        logger.info("Running validation...")
        valid_loss, valid_acc = validate(model, validloader, criterion, device)

        logger.info(f"Validation loss: {valid_loss:.4f}")
        logger.info(f"Validation accuracy: {valid_acc:.4f}")

        # Checkpointing strategy:
        # Save only the best-performing model on validation accuracy.
        # This mirrors common production workflows (model selection by held-out metric).
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            ckpt_path = models_dir / "eurosat_best.pth"

            # Save minimal but sufficient metadata to reload the model:
            # - model_name + num_classes for architecture reconstruction
            # - state_dict for weights
            torch.save(
                {
                    "model_name": config.model_name,
                    "num_classes": config.num_classes,
                    "state_dict": model.state_dict(),
                },
                ckpt_path,
            )
            logger.info(f"Saved new best model (acc={best_valid_acc:.4f}) → {ckpt_path}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best validation accuracy achieved: {best_valid_acc:.4f}")
    logger.info("=" * 80)


def validate(model, dataloader, criterion, device):
    """
    Validation loop used after each training epoch.

    Returns:
        (valid_loss, valid_accuracy) computed over the full validation set.

    Notes:
        - model.eval() disables training-specific behavior (e.g. dropout)
        - torch.no_grad() reduces memory usage and speeds up inference
    """
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0

    logger.debug("Validation started...")

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            # Accumulate loss/accuracy across all validation samples
            valid_loss += loss.item() * images.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    logger.debug("Validation finished")

    # Return dataset-average loss and accuracy
    return valid_loss / total, correct / total


# Entry point:
# Allows this module to be imported without starting training,
# while still supporting `python -m ...` or direct execution.
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    train(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        valid_fraction=cfg.data.valid_fraction,
        learning_rate=cfg.training.learning_rate,
        epochs=cfg.training.epochs,
        log_interval=cfg.training.log_interval,
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
    )

if __name__ == "__main__":
    main()