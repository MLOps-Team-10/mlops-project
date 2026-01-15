from dataclasses import dataclass

import hydra
import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig


@dataclass
class ModelConfig:
    """
    Configuration container for the model.

    Using a dataclass makes the configuration:
    - explicit and self documented
    - easy to serialize (for experiments, YAML or JSON configs)
    - simple to extend without touching model logic
    """

    model_name: str
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
            and applies log softmax internally for numerical stability.
        """
        return self.backbone(x)


@hydra.main(version_base=None, config_path="conf/model", config_name="resnet18")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point.

    cfg is a DictConfig loaded from conf/model/resnet18.yaml.
    We convert it to a strongly typed dataclass for type safety and clarity.
    """

    # Convert Hydra config to dataclass
    config = ModelConfig(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
    )

    model = EuroSATModel(config)

    # Simulate a small batch of RGB images (for a quick sanity check)
    x = torch.rand(2, 3, 224, 224)

    # Forward pass to verify that the model runs end to end
    logits = model(x)

    print("Output shape:", logits.shape)  # Expected: [2, num_classes]


if __name__ == "__main__":
    main()
