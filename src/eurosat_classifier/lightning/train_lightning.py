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
    Lightning training entrypoint.

    Hydra note:
    - Hydra changes the working directory to something like outputs/...
    - get_original_cwd() points back to the repo root (where you launched the command)
    """
    repo_root = Path(get_original_cwd())

    # If cfg.data.data_dir is a relative path (e.g. "data/raw/rgb"),
    # we anchor it to the repo root so it works even under Hydra outputs/.
    data_dir = str(repo_root / cfg.data.data_dir)

    dm = EuroSATDataModule(
        data_dir=data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        valid_fraction=cfg.data.valid_fraction,
    )

    model = EuroSATLitModule(
        model_name=cfg.model.model_name,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        learning_rate=cfg.training.learning_rate,
    )

    # Save best model according to validation accuracy (like your "best_valid_acc").
    ckpt = ModelCheckpoint(
        dirpath="models",
        filename="eurosat-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    # Early stopping is optional but useful in Lightning.
    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        callbacks=[ckpt, es],
        log_every_n_steps=cfg.training.log_interval,
        # If you want deterministic-ish splits/repro:
        # deterministic=True,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()