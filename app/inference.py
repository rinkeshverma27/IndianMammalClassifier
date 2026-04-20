"""
Inference helpers for the Streamlit app.
"""

from __future__ import annotations

import json
from pathlib import Path

import timm
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_CHECKPOINT_PATH = Path("models/indian_mammals_efficientnet_b0.pt")
DEFAULT_INFERENCE_CONFIG_PATH = Path("config/inference_config.yaml")
DEFAULT_INFERENCE_CONFIG = {
    "mammal_detector_threshold": 0.20,
    "classifier_unknown_threshold": 0.55,
    "top_k": 3,
}
MAMMAL_KEYWORDS = {
    "aardvark",
    "alpaca",
    "antelope",
    "armadillo",
    "badger",
    "bison",
    "boar",
    "buffalo",
    "camel",
    "cat",
    "cattle",
    "chihuahua",
    "chimpanzee",
    "colobus",
    "cougar",
    "cow",
    "coyote",
    "deer",
    "dingo",
    "dog",
    "donkey",
    "elephant",
    "ferret",
    "fox",
    "gazelle",
    "gibbon",
    "goat",
    "gorilla",
    "hamster",
    "hare",
    "hippopotamus",
    "hog",
    "horse",
    "hyaena",
    "hyena",
    "ibex",
    "impala",
    "jackal",
    "jaguar",
    "kit",
    "koala",
    "leopard",
    "lion",
    "llama",
    "lynx",
    "macaque",
    "mammal",
    "marmoset",
    "marten",
    "meerkat",
    "mink",
    "mole",
    "mongoose",
    "monkey",
    "mouse",
    "otter",
    "ox",
    "panda",
    "pig",
    "platypus",
    "polecat",
    "porcupine",
    "primate",
    "puma",
    "rabbit",
    "ram",
    "red wolf",
    "rhinoceros",
    "rodent",
    "sheep",
    "sloth",
    "sorrel",
    "squirrel",
    "stallion",
    "swine",
    "tiger",
    "weasel",
    "wild boar",
    "wolf",
    "wombat",
    "zebra",
}


def build_transform(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_inference_config(path: Path = DEFAULT_INFERENCE_CONFIG_PATH) -> dict:
    config = DEFAULT_INFERENCE_CONFIG.copy()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        config.update(loaded)
    return config


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


def load_mammal_detector() -> tuple[torch.nn.Module, T.Compose, set[int]]:
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    transform = weights.transforms()

    mammal_indices = {
        idx
        for idx, category in enumerate(weights.meta["categories"])
        if any(keyword in category.lower() for keyword in MAMMAL_KEYWORDS)
    }
    return model, transform, mammal_indices


def load_species_info(path: Path = Path("app/species_info.json")) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@torch.no_grad()
def mammal_filter_score(
    image: Image.Image,
    detector_model: torch.nn.Module,
    detector_transform: T.Compose,
    mammal_indices: set[int],
) -> float:
    image = image.convert("RGB")
    tensor = detector_transform(image).unsqueeze(0)
    logits = detector_model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    return float(probs[list(mammal_indices)].sum().item())


@torch.no_grad()
def classify_image(
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


def analyze_image(
    image: Image.Image,
    classifier_model: torch.nn.Module,
    classes: list[str],
    classifier_config: dict,
    detector_model: torch.nn.Module,
    detector_transform: T.Compose,
    mammal_indices: set[int],
    inference_config: dict,
) -> dict:
    mammal_score = mammal_filter_score(
        image=image,
        detector_model=detector_model,
        detector_transform=detector_transform,
        mammal_indices=mammal_indices,
    )

    if mammal_score < inference_config["mammal_detector_threshold"]:
        return {
            "status": "rejected_non_mammal",
            "mammal_score": mammal_score,
            "predictions": [],
            "message": "This image does not appear to be a mammal.",
        }

    predictions = classify_image(
        image=image,
        model=classifier_model,
        classes=classes,
        img_size=classifier_config.get("img_size", 224),
        top_k=inference_config["top_k"],
    )
    top_prediction = predictions[0]

    if top_prediction["confidence"] < inference_config["classifier_unknown_threshold"]:
        return {
            "status": "rejected_unknown",
            "mammal_score": mammal_score,
            "predictions": predictions,
            "message": "This looks like a mammal, but not confidently one of the 14 Indian mammal classes.",
        }

    return {
        "status": "accepted",
        "mammal_score": mammal_score,
        "predictions": predictions,
        "message": "",
    }
