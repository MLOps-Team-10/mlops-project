from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from eurosat_classifier.lightning.lightning_data import EuroSATDataModule
from eurosat_classifier.lightning.lightning_module import EuroSATLitModule


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training entrypoint using PyTorch Lightning and Hydra.

    Design rationale:
    - Hydra is used to manage all hyperparameters and paths in a declarative way.
    - PyTorch Lightning handles the training loop, logging, and checkpointing.
    - Data, model, and training logic are cleanly separated (MLOps best practice).
    """

    # Hydra changes the current working directory to an experiment-specific
    # output directory (e.g. outputs/2026-01-13/10-43-12).
    # We recover the original project root to build absolute paths reliably.
    repo_root = Path(get_original_cwd())

    # Resolve dataset path relative to the project root.
    # This allows the data location to be changed exclusively via Hydra config
    # without touching any training or orchestration code.
    data_dir = str(repo_root / cfg.data.data_dir)

    # Initialize the Lightning DataModule.
    # The DataModule encapsulates:
    # - dataset loading
    # - train/validation splitting
    # - DataLoader creation
    # This ensures reproducibility and simplifies the Trainer interface.
    dm = EuroSATDataModule(
        data_dir=data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        valid_fraction=cfg.data.valid_fraction,
    )

    # Initialize the LightningModule.
    # The module contains:
    # - model architecture
    # - forward pass
    # - loss computation
    # - optimizer configuration
    # All hyperparameters are injected from Hydra config.
    model = EuroSATLitModule(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        learning_rate=cfg.training.learning_rate,
    )

    # ModelCheckpoint callback:
    # - Saves the best model according to validation accuracy.
    # - Enables easy model selection and downstream deployment.
    ckpt = ModelCheckpoint(
        dirpath="models",
        filename="eurosat-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    # EarlyStopping callback:
    # - Stops training when validation loss stops improving.
    # - Prevents overfitting and unnecessary compute usage.
    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
    )

    # PyTorch Lightning Trainer:
    # - Orchestrates the full training loop
    #shows explicit configuration of training duration and logging frequency.
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        callbacks=[ckpt, es],
        log_every_n_steps=cfg.training.log_interval,
    )

    # Start training using the provided DataModule.
    # Lightning automatically calls:
    # - dm.setup()
    # - training_step / validation_step
    # - optimizer configuration
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()