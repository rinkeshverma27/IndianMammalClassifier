"""
Create a reproducible 80/20 train/validation split for the curated mammal classes.

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

TARGET_CLASS_MAP = {
    "Asiatic Lion": "Asiatic Lion",
    "Asiatic Wildcat": "Asiatic Wildcat",
    "Bengal Fox": "Bengal Fox",
    "Chinkara": "Chinkara",
    "Chital": "Chital",
    "Four Horned Antelope": "Four Horned Antelope",
    "Golden Jackal": "Golden Jackal",
    "Honey Badger": "Honey Badger",
    "Indian Leopard": "Indian Leopard",
    "Jungle Cat": "Jungle Cat",
    "Nilgai": "Nilgai",
    "Ruddy Mongoose": "Ruddy Mongoose",
    "Striped Hyena": "Striped Hyena",
    "Wild Boar": "Wild Boar",
    "barasingha": "Barasingha",
    "barking_deer": "Barking Deer",
    "bengal_tiger": "Bengal Tiger",
    "blackbuck": "Blackbuck",
    "bonnet_macaque": "Bonnet Macaque",
    "dhole": "Dhole",
    "ganges_river_dolphin": "Ganges River Dolphin",
    "gaur": "Gaur",
    "golden_langur": "Golden Langur",
    "greater_bandicoot_rat": "Greater Bandicoot Rat",
    "hanuman_langur": "Hanuman Langur",
    "indian_elephant": "Indian Elephant",
    "indian_flying_fox": "Indian Flying Fox",
    "indian_giant_squirrel": "Indian Giant Squirrel",
    "indian_grey_mongoose": "Indian Grey Mongoose",
    "indian_palm_squirrel": "Indian Palm Squirrel",
}


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

    source_dirs = sorted(
        [
            path.name
            for path in DATA_DIR.iterdir()
            if path.is_dir() and path.name not in {"train", "val"}
        ]
    )

    skipped_dirs = [name for name in source_dirs if name not in TARGET_CLASS_MAP]
    if skipped_dirs:
        print("Skipped source folders not present in curated class map:")
        for name in skipped_dirs:
            print(f"  - {name}")
        print()

    total_train = 0
    total_val = 0
    weak_classes = []

    for source_name, display_name in TARGET_CLASS_MAP.items():
        class_dir = DATA_DIR / source_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        images = image_files(class_dir)
        if len(images) < MIN_IMAGES:
            weak_classes.append((display_name, len(images)))

        random.shuffle(images)
        n_val = max(1, int(len(images) * VAL_SPLIT))
        val_images = images[:n_val]
        train_images = images[n_val:]

        copy_subset(train_images, TRAIN_DIR / display_name)
        copy_subset(val_images, VAL_DIR / display_name)

        total_train += len(train_images)
        total_val += len(val_images)

        print(
            f"{display_name:<24} total={len(images):>3} "
            f"train={len(train_images):>3} val={len(val_images):>3}"
        )

    if weak_classes:
        print("\nWarnings:")
        for class_name, count in weak_classes:
            print(f"  - {class_name}: only {count} images")

    print("\nSplit complete.")
    print(f"Classes: {len(TARGET_CLASS_MAP)}")
    print(f"Train images: {total_train}")
    print(f"Val images:   {total_val}")


if __name__ == "__main__":
    main()
