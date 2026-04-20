"""
Inference helpers for the Streamlit app.
"""

from __future__ import annotations

import json
from pathlib import Path

import timm
import torch
import torchvision.transforms as T
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_CHECKPOINT_PATH = Path("models/indian_mammals_efficientnet_b0.pt")


def build_transform(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_checkpoint(checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH) -> tuple[torch.nn.Module, list[str], dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint at {checkpoint_path}. Train the model before running the app."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in sorted(idx_to_class)]
    config = checkpoint.get("config", {})

    model = timm.create_model(
        config.get("model_name", "efficientnet_b0"),
        pretrained=False,
        num_classes=len(classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes, config


def load_species_info(path: Path = Path("app/species_info.json")) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@torch.no_grad()
def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    classes: list[str],
    img_size: int = 224,
    top_k: int = 3,
) -> list[dict]:
    image = image.convert("RGB")
    transform = build_transform(img_size)
    tensor = transform(image).unsqueeze(0)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(classes)), dim=1)

    results = []
    for prob, idx in zip(top_probs[0].tolist(), top_indices[0].tolist()):
        results.append({"label": classes[idx], "confidence": prob})
    return results
