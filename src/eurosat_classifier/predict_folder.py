from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from torchvision import transforms

from eurosat_classifier.model import EuroSATModel, ModelConfig

# Class labels in the SAME order used during training.
# ImageFolder assigns class indices alphabetically, so this order must match exactly
# to ensure correct label decoding at inference time.
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def select_device() -> torch.device:
    """
    Select the best available device for inference.

    Priority:
    - MPS for Apple Silicon
    - CUDA for NVIDIA GPUs
    - CPU as a fallback

    Centralizing this logic ensures consistent behavior across scripts.
    """
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        logger.info("Using CUDA")
        return torch.device("cuda")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")


def build_transform() -> transforms.Compose:
    """
    Build the preprocessing pipeline.

    IMPORTANT:
    This must exactly match the preprocessing used during training.
    Any mismatch here would lead to degraded or misleading predictions.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_model(ckpt_path: Path, device: torch.device) -> EuroSATModel:
    """
    Load a trained EuroSAT model from a checkpoint.

    The checkpoint contains:
    - architecture metadata (model_name, num_classes)
    - trained weights (state_dict)

    The model is set to eval mode to disable training-specific layers
    such as dropout or batch normalization updates.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    logger.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Rebuild model configuration from checkpoint metadata
    config = ModelConfig(
        model_name=checkpoint["model_name"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,  # weights are loaded from checkpoint
    )

    model = EuroSATModel(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def predict_image(
    model: EuroSATModel,
    device: torch.device,
    img_path: Path,
    transform: transforms.Compose,
):
    """
    Run inference on a single image.

    Steps:
    - load image with PIL
    - enforce RGB (3 channels)
    - apply preprocessing
    - run forward pass
    - apply softmax to obtain probabilities
    - return top-1 prediction with confidence
    """
    # Load image and ensure RGB format
    img = Image.open(img_path).convert("RGB")

    # Apply preprocessing and add batch dimension
    x = transform(img).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_prob, top_idx = torch.max(probs, dim=0)
    top_idx_int = int(top_idx.item())
    pred_class = EUROSAT_CLASSES[top_idx_int]
    confidence = top_prob.item()

    return pred_class, confidence


def predict_folder(folder_path: str = "data/test"):
    """
    Run inference on all images in a folder.

    This function:
    - loads the trained model once
    - iterates over all image files in the folder
    - logs predictions in a human-readable format

    This design is suitable for batch offline inference or demos.
    """
    device = select_device()
    transform = build_transform()
    model = load_model(Path("models/eurosat_best.pth"), device)

    folder = Path(folder_path)
    image_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    if not image_paths:
        logger.error(f"No images found in {folder}")
        return

    logger.info(f"Found {len(image_paths)} image(s) in {folder}")

    for img_path in image_paths:
        pred_class, conf = predict_image(model, device, img_path, transform)
        logger.info(f"{img_path.name:25} → {pred_class:15} (conf = {conf:.3f})")


# Script entry point.
# Allows this file to be imported without triggering inference,
# while still supporting direct execution.
if __name__ == "__main__":
    predict_folder("data/test")
