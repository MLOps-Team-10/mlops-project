from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from eurosat_classifier.model import EuroSATModel, ModelConfig


class EuroSATLitModule(pl.LightningModule):
    """
    PyTorch LightningModule that wraps the plain PyTorch EuroSATModel.

    Engineering rationale:
    - Centralizes training/validation logic, optimizer, and metrics.
    - Enables Trainer features (checkpointing, early stopping, etc.).
    - Stores all relevant hyperparameters in checkpoints for reproducibility.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()

        # Save all constructor args (including learning_rate) into the checkpoint.
        # These are accessible via self.hparams and are critical for reproducibility.
        self.save_hyperparameters()

        cfg = ModelConfig(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.model = EuroSATModel(cfg)

        # Separate metric instances avoid accidental state sharing across phases.
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (CrossEntropyLoss expects logits, not probabilities)."""
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One training step on a batch."""
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)

        # TorchMetrics MulticlassAccuracy can take class indices directly.
        preds = logits.argmax(dim=-1)
        acc = self.train_acc(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """One validation step on a batch."""
        x, y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = self.val_acc(preds, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer for the Trainer."""
        return torch.optim.Adam(self.parameters(), lr=float(self.hparams.learning_rate))
