"""
Train an EfficientNet-B0 classifier on the prepared train/val split.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


DEFAULT_CONFIG = {
    "model_name": "efficientnet_b0",
    "img_size": 224,
    "batch_size": 16,
    "epochs": 8,
    "lr_head": 1e-3,
    "lr_backbone": 1e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "patience": 3,
    "train_dir": "data/train",
    "val_dir": "data/val",
    "checkpoint_path": "models/indian_mammals_efficientnet_b0.pt",
    "class_names_path": "models/class_names.json",
    "history_path": "models/training_history.json",
}
CONFIG_PATH = Path("config/train_config.yaml")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def pil_rgb_loader(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def load_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        unknown_keys = sorted(set(loaded) - set(DEFAULT_CONFIG))
        if unknown_keys:
            raise KeyError(f"Unknown config keys in {CONFIG_PATH}: {', '.join(unknown_keys)}")
        config.update(loaded)
    return config


def build_transforms(img_size: int) -> tuple[T.Compose, T.Compose]:
    train_transform = T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def build_dataloaders(config: dict) -> tuple[ImageFolder, ImageFolder, DataLoader, DataLoader]:
    train_transform, val_transform = build_transforms(config["img_size"])
    train_dataset = ImageFolder(
        config["train_dir"], transform=train_transform, loader=pil_rgb_loader
    )
    val_dataset = ImageFolder(
        config["val_dir"], transform=val_transform, loader=pil_rgb_loader
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"] * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataset, val_dataset, train_loader, val_loader


def build_model(config: dict, num_classes: int) -> nn.Module:
    return timm.create_model(config["model_name"], pretrained=True, num_classes=num_classes)


def split_params(model: nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    backbone, head = [], []
    for name, param in model.named_parameters():
        if "classifier" in name:
            head.append(param)
        else:
            backbone.append(param)
    return backbone, head


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Train", leave=False)
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc="Val", leave=False)
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

    return running_loss / total, correct / total


def main() -> None:
    config = load_config()
    train_dir = Path(config["train_dir"])
    val_dir = Path(config["val_dir"])
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Missing data/train or data/val. Run `python3 scripts/prepare_dataset.py` first."
        )

    Path("models").mkdir(parents=True, exist_ok=True)
    print(f"Loaded config from: {CONFIG_PATH if CONFIG_PATH.exists() else 'defaults'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu")

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(config)
    model = build_model(config, num_classes=len(train_dataset.classes)).to(device)

    backbone_params, head_params = split_params(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config["lr_backbone"]},
            {"params": head_params, "lr": config["lr_head"]},
        ],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{config['epochs']} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={time.time() - start:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": train_dataset.class_to_idx,
                    "config": config,
                    "val_acc": val_acc,
                },
                config["checkpoint_path"],
            )
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print("Early stopping triggered.")
                break

    idx_to_class = {idx: name for name, idx in train_dataset.class_to_idx.items()}
    with open(config["class_names_path"], "w", encoding="utf-8") as handle:
        json.dump(idx_to_class, handle, indent=2, ensure_ascii=True)
    with open(config["history_path"], "w", encoding="utf-8") as handle:
        json.dump(
            {
                "history": history,
                "best_val_acc": best_val_acc,
                "num_classes": len(train_dataset.classes),
                "train_images": len(train_dataset),
                "val_images": len(val_dataset),
            },
            handle,
            indent=2,
            ensure_ascii=True,
        )

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved checkpoint to {config['checkpoint_path']}")


if __name__ == "__main__":
    main()
