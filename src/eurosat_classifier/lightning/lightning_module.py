import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from eurosat_classifier.model import EuroSATModel, ModelConfig


class EuroSATLitModule(pl.LightningModule):
    """
    PyTorch LightningModule that wraps the plain PyTorch EuroSATModel.

    Why this exists (MLOps/engineering rationale):
    - Encapsulates training/validation logic in a single, testable unit.
    - Standardizes logging (loss/metrics) and optimizer configuration.
    - Enables easy integration with Trainer features (checkpointing, early stopping, etc.).
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
    ) -> None:
        super().__init__()

        # Saves constructor arguments into the checkpoint and makes them available via self.hparams.
        # This is crucial for reproducibility and for re-loading the model with the same config.
        self.save_hyperparameters()

        # Build a strongly-typed config object for the underlying model.
        # Keeping "model config" separate from training config improves modularity and clarity.
        cfg = ModelConfig(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.model = EuroSATModel(cfg)

        # Metrics are stateful in TorchMetrics.
        # Defining separate instances for train/val avoids accidental state mixing.
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass used by Lightning for inference-like calls.

        Returns raw logits (no softmax) because:
        - CrossEntropyLoss expects logits directly.
        - Softmax is only needed for probability-based metrics or post-processing.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Runs one optimization step on a single batch.

        Lightning handles boilerplate for us:
        - device placement of the batch
        - gradient zeroing
        - backward pass
        - optimizer stepping (unless using manual optimization)
        """
        x, y = batch
        logits = self(x)

        # Classification loss: expects logits and integer class labels.
        loss = F.cross_entropy(logits, y)

        # Convert logits to probabilities for accuracy computation.
        # (Alternatively, many TorchMetrics accept logits, but this is explicit and easy to reason about.)
        probs = logits.softmax(dim=-1)
        acc = self.train_acc(probs, y)

        # Logging strategy:
        # - train_loss: logged per step and aggregated per epoch (useful for debugging + monitoring)
        # - train_acc: typically aggregated per epoch (less noisy than per-step)
        # prog_bar=True shows values in the Trainer progress bar.
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # Lightning uses the returned loss to run backprop automatically.
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Runs a forward-only validation step on a single batch.

        Notes:
        - No optimizer updates happen here.
        - Metrics are aggregated across the full validation epoch by TorchMetrics + Lightning logging.
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        probs = logits.softmax(dim=-1)
        acc = self.val_acc(probs, y)

        # Log validation metrics at epoch-level (standard practice).
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Defines the optimizer(s) used by the Trainer.

        Using self.hparams.learning_rate ensures the LR is checkpointed and fully config-driven.
        This supports reproducible experiments and clean Hydra-driven sweeps.
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)