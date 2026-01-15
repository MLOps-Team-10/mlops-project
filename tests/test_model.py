import torch

from eurosat_classifier.model import ModelConfig, EuroSATModel


def test_eurosat_model_forward():
    """
    Basic test to verify that the EuroSATModel can be constructed and run
    a forward pass without errors.

    The test checks:
    - correct model instantiation
    - correct output type and shape
    - compatibility with typical EuroSAT input dimensions
    """

    # Create a small configuration
    config = ModelConfig(
        model_name="resnet18",
        num_classes=10,
        pretrained=False,  # disable pretrained for faster tests
    )

    # Instantiate the model
    model = EuroSATModel(config)

    # Create a fake batch of RGB images
    # Shape: (batch_size, channels, height, width)
    x = torch.rand(4, 3, 224, 224)

    # Run a forward pass
    logits = model(x)

    # Output should be a tensor
    assert isinstance(logits, torch.Tensor)

    # Output shape should be (batch_size, num_classes)
    assert logits.shape == (4, config.num_classes)
