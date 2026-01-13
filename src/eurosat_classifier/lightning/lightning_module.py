import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from eurosat_classifier.model import EuroSATModel, ModelConfig


class EuroSATLitModule(pl.LightningModule):
    """
    LightningModule wrapper around your plain PyTorch model.

    Responsibilities:
    - define forward pass (delegates to EuroSATModel)
    - define training_step and validation_step (what happens per batch)
    - define optimizer configuration (configure_optimizers)

    Lightning then takes care of the boilerplate:
    - epoch loops
    - device placement
    - logging aggregation
    - checkpointing, etc.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
        learning_rate: float,
    ) -> None:
        super().__init__()

        # Saves init args to self.hparams, so they are:
        # - stored inside checkpoints
        # - visible in loggers
        # - reproducible across runs
        self.save_hyperparameters()

        # Build the underlying PyTorch model from your existing config + wrapper
        cfg = ModelConfig(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.model = EuroSATModel(cfg)

        # Metrics objects keep state across batches and compute epoch-level results.
        # We keep separate instances for train and val to avoid state mixing.
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass used by Lightning and also by inference code.

        Returns logits (not probabilities).
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        One training step = one batch forward + loss computation + logging.

        Lightning handles:
        - backward()
        - optimizer.step()
        - zero_grad()
        (unless you override those behaviors explicitly)
        """
        x, y = batch
        logits = self(x)

        # Cross-entropy expects logits; do NOT apply softmax before loss.
        loss = F.cross_entropy(logits, y)

        # Accuracy:
        # torchmetrics can also accept logits directly, so softmax is optional.
        acc = self.train_acc(logits.softmax(dim=-1), y)

        # self.log integrates with Lightning's logging system:
        # - on_step=True logs per step (batch)
        # - on_epoch=True aggregates across the epoch
        # - prog_bar=True shows it in the progress bar
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        One validation step = forward + loss + metrics logging.
        No optimizer updates happen here.
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Same note: softmax is optional for torchmetrics, but OK to keep.
        acc = self.val_acc(logits.softmax(dim=-1), y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Define the optimizer (and optionally schedulers).

        Lightning will call this once and manage stepping automatically.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)