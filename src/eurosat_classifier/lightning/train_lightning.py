from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from eurosat_classifier.lightning.lightning_data import EuroSATDataModule
from eurosat_classifier.lightning.lightning_module import EuroSATLitModule


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Lightning training entrypoint.

    Compared to the manual loop, this script:
    - delegates training loop + device placement to Lightning's Trainer
    - uses callbacks for checkpointing and early stopping
    - keeps configuration controlled by Hydra YAMLs
    """

    # Hydra changes the working directory to something like outputs/YYYY-MM-DD/...
    # get_original_cwd() gives you the repo root (where you launched the script).
    repo_root = Path(get_original_cwd())

    # Make data_dir absolute so it works regardless of Hydra run directory
    data_dir = str(repo_root / cfg.data.data_dir)

    # DataModule encapsulates:
    # - dataset creation
    # - train/val split
    # - dataloaders
    dm = EuroSATDataModule(
        data_dir=data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        valid_fraction=cfg.data.valid_fraction,
    )

    # LightningModule encapsulates:
    # - model forward
    # - training_step / validation_step
    # - optimizer definition
    model = EuroSATLitModule(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        learning_rate=cfg.training.learning_rate,
    )

    # Checkpoint callback:
    # - monitor val_acc and keep only best checkpoint
    # - saves under "models/" (relative to default_root_dir below)
    ckpt = ModelCheckpoint(
        dirpath="models",
        filename="eurosat-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    # Early stopping callback:
    # - stop if val_loss does not improve for `patience` epochs
    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
    )

    # Trainer manages everything:
    # - epochs/steps loop
    # - device placement (cpu/cuda/mps)
    # - logging and callbacks
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        callbacks=[ckpt, es],
        log_every_n_steps=cfg.training.log_interval,
        accelerator="auto",
        devices="auto",
        default_root_dir=str(repo_root),  # keep outputs/models relative to repo root
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()