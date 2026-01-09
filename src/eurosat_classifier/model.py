from dataclasses import dataclass

import timm
import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """
    Configuration container for the model.

    Using a dataclass makes the configuration:
    - explicit and self-documented
    - easy to serialize (e.g. for experiments, YAML/JSON configs)
    - simple to extend without touching model logic
    """
    model_name: str = "resnet18"
    num_classes: int = 10
    pretrained: bool = True


class EuroSATModel(nn.Module):
    """
    EuroSAT image classifier built on top of a timm backbone.

    This module acts as a thin wrapper around a pretrained vision backbone,
    delegating feature extraction and classification to timm while keeping
    the interface compatible with standard PyTorch training pipelines.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # Store configuration for reproducibility and experiment tracking
        self.config = config

        # Create the backbone model using timm.
        # - pretrained controls whether ImageNet weights are loaded
        # - num_classes replaces the final classification head
        # - in_chans=3 specifies RGB input images
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            num_classes=config.num_classes,
            in_chans=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input batch of images with shape
                              (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Raw output logits with shape
                          (batch_size, num_classes)

        Note:
            Softmax is intentionally NOT applied here.
            This allows the use of nn.CrossEntropyLoss, which expects logits
            and applies log-softmax internally for numerical stability.
        """
        return self.backbone(x)


# The following block is executed only when running this file directly.
# It acts as a lightweight sanity check for the model definition.
#
# This is useful in MLOps pipelines to:
# - validate model wiring
# - catch shape mismatches early
# - ensure that refactoring does not silently break the forward pass
if __name__ == "__main__":
    cfg = ModelConfig()
    model = EuroSATModel(cfg)

    # Simulate a small batch of RGB images (e.g. from EuroSAT)
    # Shape: (batch_size=2, channels=3, height=224, width=224)
    x = torch.rand(2, 3, 224, 224)

    # Forward pass to verify that the model runs end-to-end
    logits = model(x)

    # Expected output shape: (2, num_classes)
    print("Output shape:", logits.shape)  # [2, 10]