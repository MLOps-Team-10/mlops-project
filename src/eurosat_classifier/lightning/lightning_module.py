import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from eurosat_classifier.model import EuroSATModel, ModelConfig


class EuroSATLitModule(pl.LightningModule):
    """
    LightningModule wrapper around your existing EuroSATModel.

    Minimal changes compared to your manual training loop:
    - training_step replaces the inner batch loop
    - validation_step replaces validate(...)
    - configure_optimizers replaces explicit optimizer creation
    - logging is done via self.log(...)
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
        learning_rate: float,
    ) -> None:
        super().__init__()

        # Saves constructor args into the checkpoint automatically.
        # This is useful to reproduce runs and reload the model.
        self.save_hyperparameters()

        cfg = ModelConfig(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.model = EuroSATModel(cfg)

        # Accuracy metrics (TorchMetrics integrates nicely with Lightning).
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns logits (no softmax)."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        One training step = one batch.

        Lightning handles:
        - moving batch to device
        - zeroing gradients
        - backward()
        - optimizer.step()
        """
        x, y = batch
        logits = self(x)

        # CrossEntropy expects raw logits.
        loss = F.cross_entropy(logits, y)

        # TorchMetrics for multiclass accuracy expects probabilities or labels.
        probs = logits.softmax(dim=-1)
        acc = self.train_acc(probs, y)

        # Log to progress bar and to the logger.
        # - on_step: logs per batch
        # - on_epoch: aggregates over the epoch
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step run on the val_dataloader."""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        probs = logits.softmax(dim=-1)
        acc = self.val_acc(probs, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Defines the optimizer (same as your manual Adam)."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)