"""
Create a reproducible 80/20 train/validation split for the 14 uploaded classes.

Source images are expected as folders directly under data/.
Generated splits are written to data/train and data/val.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path


DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
VAL_SPLIT = 0.20
SEED = 42
MIN_IMAGES = 80
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

TARGET_CLASSES = [
    "Asiatic Lion",
    "Asiatic Wildcat",
    "Bengal Fox",
    "Chinkara",
    "Chital",
    "Four Horned Antelope",
    "Golden Jackal",
    "Honey Badger",
    "Indian Leopard",
    "Jungle Cat",
    "Nilgai",
    "Ruddy Mongoose",
    "Striped Hyena",
    "Wild Boar",
]


def image_files(folder: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
        ]
    )


def reset_output_dirs() -> None:
    for folder in (TRAIN_DIR, VAL_DIR):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)


def copy_subset(files: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, destination / src.name)


def main() -> None:
    random.seed(SEED)
    reset_output_dirs()

    print("=" * 60)
    print("DATASET PREPARATION REPORT")
    print("=" * 60)

    total_train = 0
    total_val = 0

    for class_name in TARGET_CLASSES:
        class_dir = DATA_DIR / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        images = image_files(class_dir)
        if len(images) < MIN_IMAGES:
            print(f"WARNING: {class_name} has only {len(images)} images")

        random.shuffle(images)
        n_val = max(1, int(len(images) * VAL_SPLIT))
        val_images = images[:n_val]
        train_images = images[n_val:]

        copy_subset(train_images, TRAIN_DIR / class_name)
        copy_subset(val_images, VAL_DIR / class_name)

        total_train += len(train_images)
        total_val += len(val_images)

        print(
            f"{class_name:<24} total={len(images):>3} "
            f"train={len(train_images):>3} val={len(val_images):>3}"
        )

    print("\nSplit complete.")
    print(f"Classes: {len(TARGET_CLASSES)}")
    print(f"Train images: {total_train}")
    print(f"Val images:   {total_val}")


if __name__ == "__main__":
    main()
