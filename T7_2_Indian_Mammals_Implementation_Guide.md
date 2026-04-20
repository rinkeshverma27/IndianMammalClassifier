# T7.2 — Indian Mammals Identifier: Complete Technical Implementation Guide

> **Assignment:** SMAI Assignment 3 — T7.2 Indian Mammals Identifier  
> **Stack:** EfficientNet-B0 (timm) · PyTorch · Streamlit · Google Gemini API  
> **Target:** 46 Indian mammal species · Top-3 predictions · IUCN status · QnA chatbot

---

## Table of Contents

1. [Project Architecture Overview](#1-project-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Environment Setup](#3-environment-setup)
4. [Stage 1: Data Acquisition & Curation](#4-stage-1-data-acquisition--curation)
5. [Stage 2: Web Scraping for Gap Species](#5-stage-2-web-scraping-for-gap-species)
6. [Stage 3: Dataset Validation & Splitting](#6-stage-3-dataset-validation--splitting)
7. [Stage 4: Model Training (EfficientNet-B0)](#7-stage-4-model-training-efficientnet-b0)
8. [Stage 5: Metadata & Knowledge Base Generation](#8-stage-5-metadata--knowledge-base-generation)
9. [Stage 6: Streamlit App Development](#9-stage-6-streamlit-app-development)
10. [Stage 7: LLM-Powered QnA Extension](#10-stage-7-llm-powered-qna-extension)
11. [Stage 8: Testing & Edge Cases](#11-stage-8-testing--edge-cases)
12. [Stage 9: Deployment (HuggingFace Spaces)](#12-stage-9-deployment-huggingface-spaces)
13. [Report Writing Guide](#13-report-writing-guide)
14. [Appendix: 46 Species List](#14-appendix-46-species-list)

---

## 1. Project Architecture Overview

```
User uploads photo
        │
        ▼
┌──────────────────┐
│  Image Preprocessor│  ← torchvision transforms (224×224, normalize)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  EfficientNet-B0 │  ← Fine-tuned on 46 Indian mammal classes
│  (timm)          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Top-3 Softmax   │  ← torch.topk(logits, k=3)
│  Predictions     │
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌──────────────────────┐
│species │  │   Gemini API         │
│_info   │  │   (QnA + Summary)    │
│.json   │  │                      │
└───┬────┘  └──────────┬───────────┘
    │                  │
    └────────┬──────────┘
             │
             ▼
┌──────────────────────────────┐
│  Streamlit UI                │
│  • Top-3 bar chart           │
│  • IUCN badge + description  │
│  • Fun fact                  │
│  • QnA chat window           │
└──────────────────────────────┘
```

**Why EfficientNet-B0 over CLIP?**

| Factor | EfficientNet-B0 (fine-tuned) | CLIP (zero-shot) |
|---|---|---|
| Accuracy (46 specific Indian species) | ~85–92% after 5 epochs | ~40–60% (struggles with subspecies) |
| Inference speed | ~20ms/image on CPU | ~80ms/image on CPU |
| Training cost | ~15 min on Colab T4 | No training |
| Confidence calibration | Good (softmax) | Poor (cosine sim not = probability) |
| Best for | Specific closed-set classification | Open-world discovery |

EfficientNet-B0 wins here because we have a fixed set of 46 Indian mammals and enough images per class. CLIP is better when you have zero images for training.

---

## 2. Repository Structure

```
indian-mammals-identifier/
│
├── data/
│   ├── raw/                    # Downloaded from Kaggle (untouched)
│   │   └── mammals-image-classification/
│   ├── train/                  # 46 class folders, ~80% images
│   ├── val/                    # 46 class folders, ~20% images
│   └── scraper_output/         # Web-scraped images for gap species
│
├── notebooks/
│   └── train_efficientnet.ipynb  # Full Colab training notebook
│
├── scripts/
│   ├── filter_and_organize.py  # Stage 1: Kaggle dataset filtering
│   ├── scrape_gap_species.py   # Stage 2: Web scraping
│   ├── split_dataset.py        # Stage 3: Train/val split
│   └── generate_metadata.py   # Stage 5: JSON knowledge base
│
├── models/
│   └── indian_mammals_v1.pth   # Trained weights (gitignored, upload separately)
│
├── app/
│   ├── app.py                  # Main Streamlit application
│   ├── inference.py            # Model loading + prediction logic
│   ├── chatbot.py              # Gemini QnA module
│   └── species_info.json       # Pre-generated metadata
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 3. Environment Setup

### 3.1 Local Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install all dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm streamlit google-generativeai pillow requests \
            beautifulsoup4 selenium webdriver-manager tqdm scikit-learn \
            matplotlib seaborn pandas numpy

# Save requirements
pip freeze > requirements.txt
```

### 3.2 Colab Setup (for training)

```python
# Cell 1 — Install
!pip install timm -q
!pip install kaggle -q

# Cell 2 — Mount Drive (to save model weights persistently)
from google.colab import drive
drive.mount('/content/drive')

# Cell 3 — Kaggle API (upload your kaggle.json first)
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content'
!mkdir -p /content/.kaggle
!cp kaggle.json /content/.kaggle/
!chmod 600 /content/.kaggle/kaggle.json
```

---

## 4. Stage 1: Data Acquisition & Curation

### 4.1 Download Kaggle Dataset

```bash
# From terminal (with kaggle CLI configured)
kaggle datasets download -d asaniczka/mammals-image-classification -p data/raw/
cd data/raw && unzip mammals-image-classification.zip
```

The dataset contains 45 mammal classes with ~1,000 images each. We need to check which ones are Indian species.

### 4.2 Filter Script — `scripts/filter_and_organize.py`

This script reads the Kaggle dataset, identifies Indian species from our approved list, and copies them into `data/train/`.

```python
"""
filter_and_organize.py
Copies relevant Indian mammal folders from Kaggle dataset into data/train/
"""

import os
import shutil
from pathlib import Path

# ── 1. Define your 46 target Indian species ──────────────────────────────────
# Keys = exact folder names in the Kaggle dataset
# Values = cleaned display names for your app
KAGGLE_TO_DISPLAY = {
    "Bengal Tiger":          "Bengal Tiger",
    "Snow Leopard":          "Snow Leopard",
    "Indian Leopard":        "Indian Leopard",
    "Asiatic Lion":          "Asiatic Lion",
    "Clouded Leopard":       "Clouded Leopard",
    "Cheetah":               "Cheetah",           # historically Indian
    "Indian Elephant":       "Indian Elephant",   # Gap species — scraped
    "Gaur":                  "Gaur",
    "Indian Rhinoceros":     "Indian Rhinoceros",
    "Blackbuck":             "Blackbuck",          # Gap species — scraped
    "Chinkara":              "Chinkara",
    "Nilgai":                "Nilgai",
    "Sambar Deer":           "Sambar Deer",
    "Spotted Deer":          "Spotted Deer",
    "Swamp Deer":            "Swamp Deer",
    "Musk Deer":             "Musk Deer",
    "Indian Bison":          "Indian Bison",
    "Wild Water Buffalo":    "Wild Water Buffalo",
    "Banteng":               "Banteng",
    "Indian Wild Ass":       "Indian Wild Ass",
    "Dhole":                 "Dhole",
    "Indian Wolf":           "Indian Wolf",
    "Striped Hyena":         "Striped Hyena",
    "Indian Fox":            "Indian Fox",
    "Golden Jackal":         "Golden Jackal",
    "Bengal Fox":            "Bengal Fox",
    "Sloth Bear":            "Sloth Bear",
    "Himalayan Brown Bear":  "Himalayan Brown Bear",
    "Sun Bear":              "Sun Bear",
    "Red Panda":             "Red Panda",
    "Indian Pangolin":       "Indian Pangolin",
    "Giant Squirrel":        "Giant Squirrel",
    "Indian Flying Squirrel":"Indian Flying Squirrel",
    "Himalayan Marmot":      "Himalayan Marmot",
    "Indian Porcupine":      "Indian Porcupine",
    "Nilgiri Tahr":          "Nilgiri Tahr",
    "Himalayan Tahr":        "Himalayan Tahr",
    "Markhor":               "Markhor",
    "Ibex":                  "Ibex",
    "Gangetic Dolphin":      "Gangetic Dolphin",
    "Irrawaddy Dolphin":     "Irrawaddy Dolphin",
    "Dugong":                "Dugong",
    "Indian Grey Mongoose":  "Indian Grey Mongoose",
    "Common Palm Civet":     "Common Palm Civet",
    "Binturong":             "Binturong",
    "Fishing Cat":           "Fishing Cat",
}

RAW_DIR  = Path("data/raw/mammals-image-classification")
TRAIN_DIR = Path("data/train")
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

found, missing = [], []

for kaggle_name, display_name in KAGGLE_TO_DISPLAY.items():
    # Kaggle folder names may have different capitalisation — do a case-insensitive search
    matches = [d for d in RAW_DIR.iterdir()
               if d.is_dir() and d.name.lower() == kaggle_name.lower()]
    
    if matches:
        src = matches[0]
        dst = TRAIN_DIR / display_name
        if not dst.exists():
            shutil.copytree(src, dst)
        found.append(display_name)
        print(f"✅  {display_name} — {len(list(dst.glob('*')))} images")
    else:
        missing.append(kaggle_name)
        print(f"❌  MISSING: {kaggle_name} — needs web scraping")

print(f"\nFound: {len(found)}/46    Missing (need scraping): {len(missing)}")
print("Missing list:", missing)
```

Run it:
```bash
python scripts/filter_and_organize.py
```

Make note of which species are printed as `MISSING` — these go to the scraper in Stage 2.

---

## 5. Stage 2: Web Scraping for Gap Species

### 5.1 Strategy — Why Two Methods

We use **two scraping methods** in sequence:

| Method | Best For | Speed | Risk |
|---|---|---|---|
| Wikimedia Commons API | High-quality, labelled, free-use images | Slow (~2 img/sec) | Low — no bot detection |
| Bing Image Search (Selenium) | Fills gaps quickly | Fast | Medium — use delays |

**Always prefer Wikimedia first.** It gives CC-licensed images which are legally safe for training.

### 5.2 Method A — Wikimedia Commons API Scraper (Safe, Preferred)

```python
"""
scrape_gap_species.py — Method A: Wikimedia Commons API
No browser needed. Respects rate limits. Legal for ML training.
"""

import requests
import time
import os
from pathlib import Path
from urllib.parse import quote

# ── Configuration ─────────────────────────────────────────────────────────────
GAP_SPECIES = [
    ("Indian Elephant",  "Indian elephant"),        # Wikimedia category name
    ("Blackbuck",        "Blackbuck"),
    ("Gangetic Dolphin", "Ganges river dolphin"),
    # Add more gap species here
]

OUTPUT_DIR = Path("data/train")
IMAGES_PER_SPECIES = 150   # Target per gap species
DELAY_SECONDS = 0.5        # Be polite to Wikimedia servers

# ── Wikimedia Commons API ─────────────────────────────────────────────────────
def get_wikimedia_images(category_name: str, limit: int = 150) -> list[str]:
    """
    Fetches image URLs from a Wikimedia Commons category.
    Uses the MediaWiki API — no browser, no bot detection.
    """
    BASE_URL = "https://commons.wikimedia.org/w/api.php"
    
    params = {
        "action":      "query",
        "list":        "categorymembers",
        "cmtitle":     f"Category:{category_name}",
        "cmtype":      "file",
        "cmlimit":     min(limit, 500),   # API max is 500 per request
        "format":      "json",
        "cmcontinue":  None,
    }
    
    image_titles = []
    
    while len(image_titles) < limit:
        if params["cmcontinue"] is None:
            params.pop("cmcontinue")
        
        resp = requests.get(BASE_URL, params=params, 
                           headers={"User-Agent": "IndianMammalResearch/1.0"})
        data = resp.json()
        
        members = data.get("query", {}).get("categorymembers", [])
        image_titles.extend([m["title"] for m in members 
                             if m["title"].lower().endswith((".jpg", ".jpeg", ".png"))])
        
        # Handle pagination
        if "continue" in data:
            params["cmcontinue"] = data["continue"]["cmcontinue"]
        else:
            break
        
        time.sleep(DELAY_SECONDS)
    
    return image_titles[:limit]


def get_image_url(file_title: str) -> str | None:
    """Resolves a Wikimedia file title to a direct download URL."""
    BASE_URL = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action":  "query",
        "titles":  file_title,
        "prop":    "imageinfo",
        "iiprop":  "url",
        "iiurlwidth": 800,   # Request 800px wide version (good for training)
        "format":  "json",
    }
    resp = requests.get(BASE_URL, params=params,
                        headers={"User-Agent": "IndianMammalResearch/1.0"})
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        info = page.get("imageinfo", [])
        if info:
            return info[0].get("thumburl") or info[0].get("url")
    return None


def download_image(url: str, save_path: Path) -> bool:
    """Downloads a single image. Returns True on success."""
    try:
        resp = requests.get(url, timeout=15,
                           headers={"User-Agent": "IndianMammalResearch/1.0"})
        if resp.status_code == 200 and len(resp.content) > 5000:  # Skip tiny files
            save_path.write_bytes(resp.content)
            return True
    except Exception as e:
        print(f"  ⚠️  Download failed: {e}")
    return False


# ── Main scraping loop ────────────────────────────────────────────────────────
def scrape_species(display_name: str, category_name: str, target: int = 150):
    save_dir = OUTPUT_DIR / display_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    existing = len(list(save_dir.glob("*.jpg"))) + len(list(save_dir.glob("*.png")))
    if existing >= target:
        print(f"⏭️   {display_name} already has {existing} images, skipping.")
        return
    
    needed = target - existing
    print(f"\n🔍  Scraping {display_name} (need {needed} more images)...")
    
    titles = get_wikimedia_images(category_name, limit=needed * 2)  # Get 2× in case some fail
    
    downloaded = 0
    for i, title in enumerate(titles):
        if downloaded >= needed:
            break
        
        url = get_image_url(title)
        if not url:
            continue
        
        ext = ".jpg" if "jpg" in url.lower() else ".png"
        filename = save_dir / f"{display_name.replace(' ', '_')}_{existing + downloaded:04d}{ext}"
        
        if download_image(url, filename):
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"  📥  {downloaded}/{needed} downloaded")
        
        time.sleep(DELAY_SECONDS)
    
    print(f"  ✅  Done: {downloaded} images saved for {display_name}")


if __name__ == "__main__":
    for display_name, category_name in GAP_SPECIES:
        scrape_species(display_name, category_name, target=IMAGES_PER_SPECIES)
    
    print("\n🎉  All gap species scraped!")
```

### 5.3 Method B — Selenium Scraper (Fallback for Stubborn Gaps)

Use this only if Wikimedia has fewer than 50 images for a species.

```python
"""
scrape_bing_fallback.py — Selenium-based Bing Image Search
Only use when Wikimedia has insufficient images.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import requests, time, os
from pathlib import Path

def setup_driver():
    """Creates a headless Chrome driver."""
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    # Spoof user agent to avoid detection
    opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)


def scrape_bing(query: str, save_dir: Path, target: int = 100):
    """
    Scrapes Bing Images for `query`.
    
    Safety measures built-in:
      - 2–4 sec random delay between downloads (mimics human speed)
      - Max 100 images per session (avoids IP flag)
      - Skips images < 10KB (usually placeholders/icons)
      - Saves only jpg/png
    """
    import random
    
    save_dir.mkdir(parents=True, exist_ok=True)
    driver = setup_driver()
    
    search_url = f"https://www.bing.com/images/search?q={query.replace(' ', '+')}&form=HDRSC2"
    driver.get(search_url)
    time.sleep(3)
    
    # Scroll to load more images
    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
    
    # Extract image elements
    imgs = driver.find_elements(By.CSS_SELECTOR, "img.mimg")
    print(f"  Found {len(imgs)} image elements")
    
    downloaded = 0
    for img in imgs:
        if downloaded >= target:
            break
        
        url = img.get_attribute("src") or img.get_attribute("data-src")
        if not url or url.startswith("data:"):
            continue
        
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 10_000:
                ext = ".jpg"
                path = save_dir / f"bing_{downloaded:04d}{ext}"
                path.write_bytes(resp.content)
                downloaded += 1
        except:
            pass
        
        time.sleep(random.uniform(1.5, 3.5))   # Random human-like delay
    
    driver.quit()
    print(f"  ✅  {downloaded} images saved")
```

### 5.4 Safe Scraping Principles

- **Always use `User-Agent` headers.** Never send requests without one.
- **Respect `robots.txt`.** Wikimedia allows scraping; Bing does not explicitly allow it — keep volumes small.
- **Delay between requests:** ≥ 0.5s for APIs, ≥ 1.5s for HTML pages.
- **Never run scrapers in parallel** (multiple threads) against the same domain.
- **Use Wikimedia Commons first** — it's legally clean, no ToS risk, images are CC-licensed.
- **Image quality filter:** skip images < 5KB (they're usually broken/icons).

---

## 6. Stage 3: Dataset Validation & Splitting

### 6.1 Validation Script — `scripts/split_dataset.py`

```python
"""
split_dataset.py
Validates the dataset and creates an 80/20 train/val split.
"""

import shutil
import random
from pathlib import Path
from collections import Counter

TRAIN_DIR = Path("data/train")
VAL_DIR   = Path("data/val")
VAL_SPLIT = 0.20
MIN_IMAGES = 80    # Warn if a class has fewer than this

VAL_DIR.mkdir(parents=True, exist_ok=True)
random.seed(42)    # Reproducible split

# ── Validate and report ───────────────────────────────────────────────────────
print("=" * 60)
print("DATASET VALIDATION REPORT")
print("=" * 60)

class_counts = {}
warnings = []

for class_dir in sorted(TRAIN_DIR.iterdir()):
    if not class_dir.is_dir():
        continue
    
    images = list(class_dir.glob("*.jpg")) + \
             list(class_dir.glob("*.jpeg")) + \
             list(class_dir.glob("*.png"))
    
    class_counts[class_dir.name] = len(images)
    
    if len(images) < MIN_IMAGES:
        warnings.append(f"⚠️  {class_dir.name}: only {len(images)} images (need {MIN_IMAGES}+)")

# Print summary
for cls, count in sorted(class_counts.items(), key=lambda x: x[1]):
    status = "✅" if count >= MIN_IMAGES else "⚠️ "
    print(f"  {status}  {cls:<35} {count:>5} images")

print(f"\nTotal classes: {len(class_counts)}")
print(f"Total images:  {sum(class_counts.values())}")

if warnings:
    print("\nWARNINGS:")
    for w in warnings:
        print(f"  {w}")

# ── Perform split ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PERFORMING 80/20 TRAIN/VAL SPLIT")
print("=" * 60)

for class_dir in sorted(TRAIN_DIR.iterdir()):
    if not class_dir.is_dir():
        continue
    
    images = list(class_dir.glob("*.jpg")) + \
             list(class_dir.glob("*.jpeg")) + \
             list(class_dir.glob("*.png"))
    
    random.shuffle(images)
    n_val = max(1, int(len(images) * VAL_SPLIT))   # At least 1 val image
    val_images = images[:n_val]
    
    # Create val subdirectory
    val_class_dir = VAL_DIR / class_dir.name
    val_class_dir.mkdir(parents=True, exist_ok=True)
    
    for img in val_images:
        shutil.move(str(img), str(val_class_dir / img.name))
    
    print(f"  {class_dir.name:<35} train={len(images)-n_val:>4}  val={n_val:>3}")

print("\n✅  Split complete!")
```

---

## 7. Stage 4: Model Training (EfficientNet-B0)

This is the core stage. Run this entirely in **Google Colab with T4 GPU** (free tier is fine).

### 7.1 Full Training Notebook — `notebooks/train_efficientnet.ipynb`

```python
# ════════════════════════════════════════════════════════
# CELL 1 — Imports & Config
# ════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json, time, copy

# ── Hyperparameters (tunable) ─────────────────────────────────
CONFIG = {
    "model_name":    "efficientnet_b0",  # ~5.3M params, mobile-friendly
    "num_classes":   46,
    "img_size":      224,
    "batch_size":    32,
    "num_epochs":    8,
    "lr_head":       1e-3,   # Learning rate for new final layer
    "lr_backbone":   1e-4,   # Learning rate for pretrained layers (10× lower)
    "weight_decay":  1e-4,
    "label_smoothing": 0.1,
    "patience":      3,      # Early stopping patience
    "train_dir":     "/content/drive/MyDrive/mammals/data/train",
    "val_dir":       "/content/drive/MyDrive/mammals/data/val",
    "save_path":     "/content/drive/MyDrive/mammals/models/indian_mammals_v1.pth",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

```python
# ════════════════════════════════════════════════════════
# CELL 2 — Data Transforms
# ════════════════════════════════════════════════════════

# ImageNet mean/std — always use these with timm pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.RandomResizedCrop(CONFIG["img_size"], scale=(0.7, 1.0)),  # Better than simple resize
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.1),          # Animals can appear at odd angles
    T.ColorJitter(
        brightness=0.3,                    # Lighting variation (day/night photos)
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    T.RandomGrayscale(p=0.05),            # Mimic B&W camera trap images
    T.RandomRotation(degrees=15),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Val transform — NO augmentation, only resize and normalize
val_transform = T.Compose([
    T.Resize(256),                         # Resize shorter edge to 256
    T.CenterCrop(CONFIG["img_size"]),       # Then center-crop to 224
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
```

```python
# ════════════════════════════════════════════════════════
# CELL 3 — Dataset & DataLoaders
# ════════════════════════════════════════════════════════

train_dataset = ImageFolder(CONFIG["train_dir"], transform=train_transform)
val_dataset   = ImageFolder(CONFIG["val_dir"],   transform=val_transform)

# Save class → index mapping (CRITICAL — needed for inference)
CLASS_TO_IDX = train_dataset.class_to_idx
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

with open("/content/drive/MyDrive/mammals/models/class_names.json", "w") as f:
    json.dump(IDX_TO_CLASS, f, indent=2)

print(f"Classes: {len(train_dataset.classes)}")
print(f"Train images: {len(train_dataset)}")
print(f"Val images:   {len(val_dataset)}")

# Check class balance
from collections import Counter
counts = Counter(train_dataset.targets)
min_cls = min(counts, key=counts.get)
max_cls = max(counts, key=counts.get)
print(f"\nMost images: {IDX_TO_CLASS[max_cls]} ({counts[max_cls]})")
print(f"Fewest images: {IDX_TO_CLASS[min_cls]} ({counts[min_cls]})")

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=2,            # 2 workers is safe on Colab
    pin_memory=True,          # Speeds up GPU transfer
    drop_last=True,           # Avoids batch norm issues with tiny last batch
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"] * 2,   # Can use larger batch for val (no grad)
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
```

```python
# ════════════════════════════════════════════════════════
# CELL 4 — Model Architecture
# ════════════════════════════════════════════════════════

def build_model(num_classes: int) -> nn.Module:
    """
    Loads EfficientNet-B0 with ImageNet weights, replaces classifier head.
    
    EfficientNet-B0 architecture:
      Stem → 7× MBConv blocks → GlobalAvgPool → Dropout(0.2) → Linear(1280, n_classes)
    
    We replace only the final Linear layer.
    """
    model = timm.create_model(
        CONFIG["model_name"],
        pretrained=True,          # ImageNet weights
        num_classes=num_classes,
    )
    
    # timm's EfficientNet has model.classifier as the final Linear
    # timm already adjusts it when you pass num_classes — but let's verify:
    print(f"Classifier layer: {model.classifier}")
    print(f"Output features: {model.classifier.out_features}")
    
    return model

model = build_model(CONFIG["num_classes"])
model = model.to(DEVICE)

# Count parameters
total_params   = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

```python
# ════════════════════════════════════════════════════════
# CELL 5 — Differential Learning Rates
# ════════════════════════════════════════════════════════
# 
# KEY CONCEPT: We use a LOWER learning rate for the pretrained backbone
# and a HIGHER learning rate for the new classifier head.
# This is called "discriminative fine-tuning" or "differential LR".
#
# Why? The ImageNet weights are already good. We don't want to 
# aggressively overwrite them — just nudge them. The new head
# (trained from random init) needs to learn fast.

# Identify backbone vs head parameters
backbone_params = [p for name, p in model.named_parameters() 
                   if "classifier" not in name]
head_params     = [p for name, p in model.named_parameters() 
                   if "classifier" in name]

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": CONFIG["lr_backbone"]},
    {"params": head_params,     "lr": CONFIG["lr_head"]},
], weight_decay=CONFIG["weight_decay"])

# Cosine annealing LR schedule — reduces LR smoothly over training
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=CONFIG["num_epochs"],
    eta_min=1e-6,
)

# Label smoothing reduces overconfidence (regularization)
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
```

```python
# ════════════════════════════════════════════════════════
# CELL 6 — Training Loop
# ════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}  "
                  f"loss={loss.item():.4f}  acc={correct/total:.3f}")
    
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    
    return total_loss / total, correct / total


# ── Training with early stopping ──────────────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc  = 0.0
best_weights  = None
patience_counter = 0

for epoch in range(1, CONFIG["num_epochs"] + 1):
    print(f"\n{'='*55}")
    print(f"EPOCH {epoch}/{CONFIG['num_epochs']}  "
          f"LR_backbone={scheduler.get_last_lr()[0]:.2e}")
    print(f"{'='*55}")
    
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_acc     = validate(model, val_loader, criterion, DEVICE)
    scheduler.step()
    elapsed = time.time() - t0
    
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.4f}")
    print(f"  Time: {elapsed:.1f}s")
    
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = copy.deepcopy(model.state_dict())
        torch.save({
            "epoch":      epoch,
            "model_state_dict": best_weights,
            "val_acc":    best_val_acc,
            "class_to_idx": CLASS_TO_IDX,
            "config":     CONFIG,
        }, CONFIG["save_path"])
        print(f"  💾  New best model saved! val_acc={best_val_acc:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  ⏳  No improvement ({patience_counter}/{CONFIG['patience']})")
    
    if patience_counter >= CONFIG["patience"]:
        print("\n⛔  Early stopping triggered!")
        break

print(f"\n🏆  Best validation accuracy: {best_val_acc:.4f}")
```

```python
# ════════════════════════════════════════════════════════
# CELL 7 — Plot Training Curves
# ════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, len(history["train_acc"]) + 1)

ax1.plot(epochs_range, history["train_acc"], "b-o", label="Train")
ax1.plot(epochs_range, history["val_acc"],   "r-o", label="Val")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epoch")
ax1.legend()
ax1.grid(True)

ax2.plot(epochs_range, history["train_loss"], "b-o", label="Train")
ax2.plot(epochs_range, history["val_loss"],   "r-o", label="Val")
ax2.set_title("Loss")
ax2.set_xlabel("Epoch")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/mammals/training_curves.png", dpi=150)
plt.show()
```

```python
# ════════════════════════════════════════════════════════
# CELL 8 — Confusion Matrix (Top-10 confused classes)
# ════════════════════════════════════════════════════════
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Get all predictions on val set
all_preds, all_labels = [], []
model.load_state_dict(best_weights)
model.eval()

with torch.no_grad():
    for images, labels in val_loader:
        logits = model(images.to(DEVICE))
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

# Show only top-10 most confused class pairs
errors = []
for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i][j] > 0:
            errors.append((cm[i][j], IDX_TO_CLASS[i], IDX_TO_CLASS[j]))
errors.sort(reverse=True)

print("Top 10 confused pairs (true → predicted):")
for count, true_cls, pred_cls in errors[:10]:
    print(f"  {true_cls} → {pred_cls}: {count} errors")
```

### 7.2 Expected Performance

After 5–8 epochs on T4 GPU:
- **Training time:** ~15–20 minutes
- **Val accuracy:** 82–90% (depends on dataset quality)
- **Inference time:** ~25ms per image on CPU

---

## 8. Stage 5: Metadata & Knowledge Base Generation

### 8.1 Generate `species_info.json` via Gemini

**Do this once.** Cache the result as JSON so the app never calls an API for metadata during inference.

```python
"""
generate_metadata.py
Calls Gemini API once to generate knowledge base for all 46 species.
Save output as app/species_info.json
"""

import google.generativeai as genai
import json
import time

# Configure Gemini
genai.configure(api_key="YOUR_GOOGLE_AI_STUDIO_KEY")   # Free at aistudio.google.com
model = genai.GenerativeModel("gemini-1.5-flash")

SPECIES_LIST = [
    "Bengal Tiger", "Snow Leopard", "Indian Leopard", "Asiatic Lion",
    "Clouded Leopard", "Indian Elephant", "Gaur", "Indian Rhinoceros",
    "Blackbuck", "Chinkara", "Nilgai", "Sambar Deer", "Spotted Deer",
    "Swamp Deer", "Musk Deer", "Wild Water Buffalo", "Indian Wild Ass",
    "Dhole", "Indian Wolf", "Striped Hyena", "Indian Fox", "Golden Jackal",
    "Bengal Fox", "Sloth Bear", "Himalayan Brown Bear", "Sun Bear",
    "Red Panda", "Indian Pangolin", "Giant Squirrel", "Indian Flying Squirrel",
    "Himalayan Marmot", "Indian Porcupine", "Nilgiri Tahr", "Himalayan Tahr",
    "Markhor", "Ibex", "Gangetic Dolphin", "Irrawaddy Dolphin", "Dugong",
    "Indian Grey Mongoose", "Common Palm Civet", "Binturong", "Fishing Cat",
    "Cheetah", "Banteng", "Indian Bison",
]

PROMPT_TEMPLATE = """
You are a wildlife biologist. For each mammal listed below, provide a JSON object
with EXACTLY these fields:
- "scientific_name": string
- "iucn_status": one of ["Extinct in Wild", "Critically Endangered", "Endangered", 
                          "Vulnerable", "Near Threatened", "Least Concern", "Data Deficient"]
- "iucn_color": hex color for status badge (red=#C0392B for CE/EN, orange=#E67E22 for VU, 
                 yellow=#F1C40F for NT, green=#27AE60 for LC)
- "habitat": short string (e.g., "Tropical forests, grasslands")
- "states_found": list of Indian states where found (e.g., ["Rajasthan", "Gujarat"])
- "description": exactly 2 sentences. Fact-based, suitable for a tourist app.
- "fun_fact": one interesting sentence.
- "diet": one of ["Carnivore", "Herbivore", "Omnivore", "Insectivore"]
- "weight_kg": typical adult weight range as string (e.g., "150–250 kg")

Return ONLY a valid JSON array. No markdown, no explanation.

Species: {species_list}
"""

def generate_metadata_batch(species_batch: list) -> list:
    prompt = PROMPT_TEMPLATE.format(species_list=json.dumps(species_batch))
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,    # Low temp for factual data
            max_output_tokens=4096,
        )
    )
    
    # Clean response (remove markdown fences if present)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    
    return json.loads(text)


# ── Process in batches of 10 ──────────────────────────────────────────────────
all_data = {}
batch_size = 10

for i in range(0, len(SPECIES_LIST), batch_size):
    batch = SPECIES_LIST[i:i + batch_size]
    print(f"Processing batch {i//batch_size + 1}: {batch}")
    
    try:
        results = generate_metadata_batch(batch)
        
        # Index by species name
        for j, species_name in enumerate(batch):
            if j < len(results):
                all_data[species_name] = results[j]
            
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parse error for batch {i//batch_size + 1}: {e}")
        print("  Retrying with individual requests...")
        for species in batch:
            try:
                result = generate_metadata_batch([species])
                all_data[species] = result[0]
                time.sleep(1)
            except:
                all_data[species] = {"error": "Could not generate metadata"}
    
    time.sleep(2)   # Avoid hitting rate limits

# Save
with open("app/species_info.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

print(f"\n✅  Generated metadata for {len(all_data)} species")
print("Saved to app/species_info.json")
```

---

## 9. Stage 6: Streamlit App Development

### 9.1 Inference Module — `app/inference.py`

```python
"""
inference.py
Handles model loading and prediction. Separated from app.py for clean architecture.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from PIL import Image
import json
from pathlib import Path
import streamlit as st   # For @st.cache_resource

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH      = Path("models/indian_mammals_v1.pth")
CLASS_NAMES_PATH = Path("models/class_names.json")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@st.cache_resource   # Load model ONCE, cache in session (fast reloads)
def load_model():
    """Loads model weights and class names. Cached by Streamlit."""
    # Load class names
    with open(CLASS_NAMES_PATH) as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}
    
    num_classes = len(idx_to_class)
    
    # Rebuild model architecture
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, idx_to_class


def predict_top3(pil_image: Image.Image) -> list[dict]:
    """
    Runs inference on a PIL image.
    Returns list of 3 dicts: [{"species": str, "confidence": float}, ...]
    sorted by confidence descending.
    """
    model, idx_to_class = load_model()
    
    # Preprocess
    tensor = INFERENCE_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0)  # Add batch dim
    
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)
    
    # Top-3 predictions
    top3_probs, top3_idxs = torch.topk(probs, k=3, dim=1)
    
    results = []
    for prob, idx in zip(top3_probs[0], top3_idxs[0]):
        results.append({
            "species":    idx_to_class[idx.item()],
            "confidence": prob.item(),
            "idx":        idx.item(),
        })
    
    return results
```

### 9.2 Chatbot Module — `app/chatbot.py`

```python
"""
chatbot.py
Wraps Gemini API for the QnA section of the app.
"""

import google.generativeai as genai
import streamlit as st
import json
from pathlib import Path

SPECIES_INFO_PATH = Path("app/species_info.json")

@st.cache_resource
def load_species_info():
    with open(SPECIES_INFO_PATH, encoding="utf-8") as f:
        return json.load(f)


def init_gemini(api_key: str):
    """Initialize Gemini client. Called once."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def build_system_prompt(detected_species: str, species_info: dict) -> str:
    """
    Builds a context-rich system prompt so Gemini answers are
    grounded in facts about the detected animal.
    """
    info = species_info.get(detected_species, {})
    
    context = f"""
You are WildGuide, a knowledgeable and friendly Indian wildlife expert chatbot embedded in 
an animal identification app.

The user has just uploaded a photo and it was identified as: **{detected_species}**

Here are the verified facts you MUST use as ground truth:
- Scientific name: {info.get('scientific_name', 'Unknown')}
- IUCN Status: {info.get('iucn_status', 'Unknown')}
- Habitat: {info.get('habitat', 'Unknown')}
- Indian states: {', '.join(info.get('states_found', ['Unknown']))}
- Diet: {info.get('diet', 'Unknown')}
- Weight: {info.get('weight_kg', 'Unknown')}
- Description: {info.get('description', '')}
- Fun fact: {info.get('fun_fact', '')}

Guidelines:
1. Answer questions about {detected_species} with enthusiasm and accuracy.
2. If asked about something unrelated to wildlife, politely redirect.
3. Keep answers concise (2-4 sentences) unless the user asks for detail.
4. For conservation questions, always mention what the user can do to help.
5. NEVER make up facts not in your training data — say "I'm not sure" instead.
"""
    return context


def get_chat_response(
    model,
    conversation_history: list,
    user_message: str,
    detected_species: str,
    species_info: dict,
) -> str:
    """
    Sends a message to Gemini with full conversation history.
    Returns the assistant's response text.
    """
    system_prompt = build_system_prompt(detected_species, species_info)
    
    # Build messages list for Gemini
    # Gemini uses role: "user" / "model" (not "assistant")
    messages = []
    
    # Inject system prompt as first user message (Gemini 1.5 Flash supports system_instruction)
    for msg in conversation_history:
        messages.append({
            "role":  msg["role"],
            "parts": [msg["content"]],
        })
    
    # Add current user message
    messages.append({
        "role":  "user",
        "parts": [user_message],
    })
    
    response = model.generate_content(
        messages,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=512,
        ),
        system_instruction=system_prompt,
    )
    
    return response.text
```

### 9.3 Main App — `app/app.py`

```python
"""
app.py
Main Streamlit application for Indian Mammals Identifier.
Run: streamlit run app/app.py
"""

import streamlit as st
from PIL import Image
import json
from pathlib import Path
import os

from inference import predict_top3
from chatbot import load_species_info, init_gemini, get_chat_response

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Mammals Identifier",
    page_icon="🐆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .iucn-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
        color: white;
        display: inline-block;
        margin: 4px 0;
    }
    .species-card {
        background: #1e1e2e;
        padding: 16px;
        border-radius: 12px;
        border-left: 4px solid #7c3aed;
        margin: 8px 0;
    }
    .chat-msg-user {
        background: #2563eb;
        color: white;
        padding: 10px 14px;
        border-radius: 12px 12px 2px 12px;
        margin: 4px 0 4px auto;
        max-width: 80%;
        width: fit-content;
    }
    .chat-msg-bot {
        background: #374151;
        color: white;
        padding: 10px 14px;
        border-radius: 12px 12px 12px 2px;
        margin: 4px auto 4px 0;
        max-width: 80%;
        width: fit-content;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/"
             "Bengal_tiger_%28Panthera_tigris_tigris%29_female_3_crop.jpg/"
             "320px-Bengal_tiger_%28Panthera_tigris_tigris%29_female_3_crop.jpg",
             use_column_width=True)
    st.title("🐾 Indian Mammals")
    st.markdown("Upload a photo of an Indian mammal to identify it and learn more.")
    st.divider()
    
    # Gemini API key input
    st.subheader("🤖 Enable AI Chat (Optional)")
    api_key = st.text_input(
        "Google AI Studio API Key",
        type="password",
        help="Free key at aistudio.google.com — enables QnA chat about the animal",
        placeholder="AIza..."
    )
    
    if api_key:
        st.success("✅ Chat enabled!")
    else:
        st.info("ℹ️ Add key to enable QnA chatbot")
    
    st.divider()
    st.caption("Model: EfficientNet-B0 fine-tuned on 46 Indian mammal species")
    st.caption("Dataset: Kaggle mammals-image-classification + web scraped")

# ── Load Resources ────────────────────────────────────────────────────────────
species_info = load_species_info()

IUCN_COLORS = {
    "Extinct in Wild":       "#1a1a1a",
    "Critically Endangered": "#C0392B",
    "Endangered":            "#E74C3C",
    "Vulnerable":            "#E67E22",
    "Near Threatened":       "#F1C40F",
    "Least Concern":         "#27AE60",
    "Data Deficient":        "#7F8C8D",
}

# ── Main Layout ───────────────────────────────────────────────────────────────
st.title("🐾 Indian Mammals Identifier")
st.markdown("*Powered by EfficientNet-B0 fine-tuned on 46 Indian species*")

col1, col2 = st.columns([1, 1.4], gap="large")

# ── LEFT COLUMN: Upload & Image Display ───────────────────────────────────────
with col1:
    uploaded_file = st.file_uploader(
        "Upload a mammal photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Works best with clear photos showing the full animal"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

# ── RIGHT COLUMN: Results ─────────────────────────────────────────────────────
with col2:
    if uploaded_file:
        with st.spinner("🔍 Identifying species..."):
            predictions = predict_top3(image)
        
        top_species = predictions[0]["species"]
        top_conf    = predictions[0]["confidence"]
        info        = species_info.get(top_species, {})
        iucn_status = info.get("iucn_status", "Data Deficient")
        iucn_color  = IUCN_COLORS.get(iucn_status, "#7F8C8D")
        
        # ── Top Prediction Header ──
        st.markdown(f"## 🦁 {top_species}")
        st.markdown(f"*{info.get('scientific_name', '')}*")
        
        badge_html = (f'<span class="iucn-badge" '
                      f'style="background:{iucn_color};">'
                      f'IUCN: {iucn_status}</span>')
        st.markdown(badge_html, unsafe_allow_html=True)
        
        # ── Confidence Bar Chart ──
        st.subheader("Top-3 Predictions")
        for pred in predictions:
            st.progress(
                pred["confidence"],
                text=f"{pred['species']}  ({pred['confidence']*100:.1f}%)"
            )
        
        # ── Info Cards ──
        st.divider()
        
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("Diet", info.get("diet", "Unknown"))
            st.metric("Weight", info.get("weight_kg", "Unknown"))
        with info_col2:
            st.metric("Habitat", info.get("habitat", "Unknown")[:30] + "...")
        
        st.markdown(f"**Found in:** {', '.join(info.get('states_found', ['Unknown'])[:5])}")
        
        st.info(f"📖 {info.get('description', 'No description available.')}")
        st.success(f"⭐ Fun Fact: {info.get('fun_fact', '')}")
        
        # ── Confidence Warning ──
        if top_conf < 0.40:
            st.warning("⚠️ Low confidence — try a clearer photo with better lighting.")
        elif top_conf < 0.70:
            st.warning("🤔 Moderate confidence — the animal might be partially obscured.")

# ── QnA CHATBOT SECTION ───────────────────────────────────────────────────────
st.divider()

if uploaded_file and predictions:
    top_species = predictions[0]["species"]
    st.subheader(f"🤖 Ask anything about the {top_species}")
    
    if not api_key:
        st.info("🔑 Add your Google AI Studio API key in the sidebar to enable this feature.")
    else:
        # Initialize chat history in session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "gemini_model" not in st.session_state:
            st.session_state.gemini_model = init_gemini(api_key)
        if "last_species" not in st.session_state:
            st.session_state.last_species = None
        
        # Reset chat if species changed (new photo uploaded)
        if st.session_state.last_species != top_species:
            st.session_state.chat_history = []
            st.session_state.last_species = top_species
            # Auto-generate opening summary
            with st.spinner("Generating summary..."):
                opening = get_chat_response(
                    st.session_state.gemini_model,
                    conversation_history=[],
                    user_message=(f"Give me a brief, engaging 3-sentence introduction to "
                                  f"the {top_species} as if you're a wildlife guide. "
                                  f"Include one conservation concern."),
                    detected_species=top_species,
                    species_info=species_info,
                )
                st.session_state.chat_history.append({
                    "role": "model",
                    "content": opening,
                })
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "model":
                    st.markdown(
                        f'<div class="chat-msg-bot">🤖 {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chat-msg-user">👤 {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
        
        # Suggested questions
        st.markdown("**Suggested questions:**")
        suggestions = [
            f"Is the {top_species} dangerous to humans?",
            f"Where can I see a {top_species} in India?",
            f"Why is the {top_species} endangered?",
            f"What does the {top_species} eat?",
        ]
        
        cols = st.columns(2)
        for i, q in enumerate(suggestions):
            if cols[i % 2].button(q, key=f"suggestion_{i}"):
                # Handle button click as user input
                st.session_state.chat_history.append({"role": "user", "content": q})
                with st.spinner("Thinking..."):
                    response = get_chat_response(
                        st.session_state.gemini_model,
                        conversation_history=st.session_state.chat_history[:-1],
                        user_message=q,
                        detected_species=top_species,
                        species_info=species_info,
                    )
                st.session_state.chat_history.append({"role": "model", "content": response})
                st.rerun()
        
        # Free-text input
        user_input = st.chat_input(f"Ask about the {top_species}...")
        
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.spinner("Thinking..."):
                response = get_chat_response(
                    st.session_state.gemini_model,
                    conversation_history=st.session_state.chat_history[:-1],
                    user_message=user_input,
                    detected_species=top_species,
                    species_info=species_info,
                )
            st.session_state.chat_history.append({"role": "model", "content": response})
            st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

elif not uploaded_file:
    st.markdown("### 📸 Upload a photo to get started")
    st.markdown("""
    **How it works:**
    1. Upload any photo of an Indian mammal
    2. The AI identifies it from 46 possible species
    3. See IUCN conservation status, habitat, and fun facts
    4. Enable the chatbot with a free Gemini API key to ask questions
    
    **Tips for best results:**
    - Use clear, well-lit photos
    - Make sure the animal is the main subject
    - Avoid heavily zoomed-in crops
    """)
```

---

## 10. Stage 7: LLM-Powered QnA Extension

### 10.1 How the QnA System Works

```
User types question
        │
        ▼
┌────────────────────────────┐
│  System Prompt Builder     │
│  • Detected species name   │
│  • All facts from JSON     │  ← "Ground truth" context
│  • Persona: WildGuide      │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Conversation History      │
│  [user, model, user, ...]  │  ← Full multi-turn context
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Gemini 1.5 Flash API      │
│  model.generate_content()  │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Response displayed in     │
│  chat UI                   │
└────────────────────────────┘
```

### 10.2 Getting the API Key (Free)

1. Visit [aistudio.google.com](https://aistudio.google.com)
2. Sign in with Google
3. Click "Get API Key" → "Create API Key"
4. Copy the key (starts with `AIza...`)
5. Free tier: 15 RPM, 1M tokens/day — plenty for a demo

### 10.3 Key Design Decisions for the QnA

**Why context-grounding?** Without it, Gemini might hallucinate facts ("The Bengal Tiger weighs 2 tons"). By injecting your `species_info.json` into the system prompt, Gemini's answers are anchored to verified data.

**Why multi-turn history?** Users naturally ask follow-up questions ("Where can I see it?" → "Is it safe?" → "How do I get there?"). We pass the full `chat_history` list each time so Gemini remembers context.

**Why auto-generate an opening summary?** Instead of waiting for the user to ask something, we immediately generate a 3-sentence intro when a species is detected. This demonstrates the feature and sets the context.

**Rate limiting / Cost:** Gemini 1.5 Flash is free up to 15 requests/minute. Since this is a demo app with occasional users, you'll never hit limits. If you want to be safe, add:

```python
import time
import functools

def rate_limit(calls_per_minute=10):
    """Simple rate limiter decorator."""
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator
```

---

## 11. Stage 8: Testing & Edge Cases

### 11.1 Test Script — `scripts/test_edge_cases.py`

```python
"""
Test the model with edge cases before deployment.
"""
import torch
import json
from PIL import Image
import requests
from io import BytesIO
from inference import predict_top3

TEST_CASES = [
    # (description, url_or_path, expected_behavior)
    ("Clear Bengal Tiger",    "https://upload.wikimedia.org/...",  "Should be Bengal Tiger >80%"),
    ("Snow Leopard - blurry", "tests/snow_leopard_blurry.jpg",     "Should give warning"),
    ("Polar Bear (non-Indian)","tests/polar_bear.jpg",             "Should pick closest: Himalayan Brown Bear"),
    ("House Cat (domestic)",  "tests/domestic_cat.jpg",            "Should pick: Fishing Cat or Clouded Leopard"),
    ("Landscape photo",       "tests/mountains.jpg",               "Should give very low confidence"),
]

for desc, source, expected in TEST_CASES:
    if source.startswith("http"):
        img = Image.open(BytesIO(requests.get(source).content))
    else:
        img = Image.open(source)
    
    preds = predict_top3(img)
    print(f"\n{desc}")
    print(f"  Expected: {expected}")
    print(f"  Got: {preds[0]['species']} ({preds[0]['confidence']*100:.1f}%)")
    print(f"  Top-3: {[(p['species'], f\"{p['confidence']*100:.0f}%\") for p in preds]}")
```

### 11.2 Behavior for Non-Indian Animals

The model will always output a prediction (it's a closed-set classifier). For non-Indian animals, document this limitation clearly:

- **Polar Bear** → will predict "Himalayan Brown Bear" (closest in appearance)
- **African Lion** → will predict "Asiatic Lion"
- **Domestic Cat** → will predict "Fishing Cat" or "Clouded Leopard"

This is **expected and correct behavior** for your report. Mention it in the Limitations section.

---

## 12. Stage 9: Deployment (HuggingFace Spaces)

### 12.1 Files to Upload to HF Spaces

```
your-hf-repo/
├── app.py              # Streamlit entry point (HF expects this at root)
├── inference.py
├── chatbot.py
├── requirements.txt
├── models/
│   ├── indian_mammals_v1.pth   # Upload via Git LFS
│   └── class_names.json
└── app/
    └── species_info.json
```

### 12.2 HF Spaces `requirements.txt`

```
torch==2.1.0
torchvision==0.16.0
timm==0.9.12
streamlit==1.32.0
google-generativeai==0.5.4
Pillow==10.2.0
```

### 12.3 Upload Model with Git LFS

```bash
# HuggingFace Spaces setup
pip install huggingface_hub
huggingface-cli login

# Create new Space
# Go to huggingface.co → New Space → SDK: Streamlit

# Clone and upload
git clone https://huggingface.co/spaces/YOUR_USERNAME/indian-mammals
cd indian-mammals
git lfs install
git lfs track "*.pth"    # Track large model files with Git LFS
git add .gitattributes

# Copy your project files in, then:
git add .
git commit -m "Initial deployment"
git push
```

### 12.4 Handle the API Key Securely on HF Spaces

In HF Spaces → Settings → Repository Secrets → Add `GEMINI_API_KEY`

Then in `app.py`:

```python
import os

# Try to get key from environment (HF Spaces secret) first,
# fall back to user input
env_key = os.environ.get("GEMINI_API_KEY", "")
api_key = env_key or st.sidebar.text_input("Google AI API Key", type="password")
```

---

## 13. Report Writing Guide

### 13.1 Suggested Outline (4–6 pages, Tier 1)

1. **Introduction** (0.5 pages) — The problem, why Indian mammals, why ML for wildlife identification.

2. **Data** (1 page) — Describe the Kaggle source, the 46 species, how you handled gap species (Wikimedia scraping), class distribution table, train/val split strategy.

3. **Method** (1 page) — EfficientNet-B0 architecture (diagram), why over CLIP, transfer learning explanation, differential LR, augmentation choices.

4. **Results** (1–1.5 pages) — Val accuracy table, training curves figure, confusion matrix for top-5 confused pairs, Top-3 accuracy (usually 5–10% higher than Top-1).

5. **Ablations** (0.5 pages) — Compare: (a) EfficientNet-B0 vs MobileNetV3 on val acc, (b) with vs without ColorJitter augmentation.

6. **App Demo** (0.5 pages) — Screenshots of the Streamlit UI. Describe the QnA flow.

7. **Limitations** (0.5 pages) — Closed-set classifier (non-Indian animals misclassified), dataset bias (most images from zoo/photography, not wild), low confidence in poor lighting.

8. **References** — Tan & Le 2019 (EfficientNet), Mohanty 2016 (inspiration), Kaggle dataset card, Gemini API docs.

### 13.2 Ablation Table Example

| Configuration | Val Top-1 Acc | Val Top-3 Acc |
|---|---|---|
| EfficientNet-B0 (ours) | **87.3%** | **96.1%** |
| MobileNetV3-Small | 82.1% | 93.4% |
| EfficientNet-B0 (no augmentation) | 83.7% | 94.2% |
| EfficientNet-B0 (no differential LR) | 85.2% | 95.0% |

---

## 14. Appendix: 46 Species List

| # | Display Name | Kaggle Available | IUCN (approx.) |
|---|---|---|---|
| 1 | Bengal Tiger | ✅ | Endangered |
| 2 | Snow Leopard | ✅ | Vulnerable |
| 3 | Indian Leopard | ✅ | Vulnerable |
| 4 | Asiatic Lion | ✅ | Endangered |
| 5 | Clouded Leopard | ✅ | Vulnerable |
| 6 | Indian Elephant | ⚠️ Scrape | Endangered |
| 7 | Gaur | ✅ | Vulnerable |
| 8 | Indian Rhinoceros | ✅ | Vulnerable |
| 9 | Blackbuck | ⚠️ Scrape | Least Concern |
| 10 | Chinkara | ✅ | Least Concern |
| 11 | Nilgai | ✅ | Least Concern |
| 12 | Sambar Deer | ✅ | Vulnerable |
| 13 | Spotted Deer | ✅ | Least Concern |
| 14 | Swamp Deer | ✅ | Vulnerable |
| 15 | Musk Deer | ✅ | Endangered |
| 16 | Wild Water Buffalo | ✅ | Endangered |
| 17 | Indian Wild Ass | ✅ | Near Threatened |
| 18 | Dhole | ✅ | Endangered |
| 19 | Indian Wolf | ⚠️ Scrape | Least Concern |
| 20 | Striped Hyena | ✅ | Near Threatened |
| 21 | Indian Fox | ⚠️ Scrape | Least Concern |
| 22 | Golden Jackal | ✅ | Least Concern |
| 23 | Bengal Fox | ⚠️ Scrape | Least Concern |
| 24 | Sloth Bear | ✅ | Vulnerable |
| 25 | Himalayan Brown Bear | ✅ | Least Concern |
| 26 | Sun Bear | ✅ | Vulnerable |
| 27 | Red Panda | ✅ | Endangered |
| 28 | Indian Pangolin | ⚠️ Scrape | Critically Endangered |
| 29 | Giant Squirrel | ⚠️ Scrape | Least Concern |
| 30 | Indian Flying Squirrel | ⚠️ Scrape | Least Concern |
| 31 | Himalayan Marmot | ⚠️ Scrape | Least Concern |
| 32 | Indian Porcupine | ⚠️ Scrape | Least Concern |
| 33 | Nilgiri Tahr | ⚠️ Scrape | Endangered |
| 34 | Himalayan Tahr | ✅ | Near Threatened |
| 35 | Markhor | ✅ | Near Threatened |
| 36 | Ibex | ✅ | Least Concern |
| 37 | Gangetic Dolphin | ⚠️ Scrape | Endangered |
| 38 | Irrawaddy Dolphin | ⚠️ Scrape | Endangered |
| 39 | Dugong | ⚠️ Scrape | Vulnerable |
| 40 | Indian Grey Mongoose | ⚠️ Scrape | Least Concern |
| 41 | Common Palm Civet | ✅ | Least Concern |
| 42 | Binturong | ✅ | Vulnerable |
| 43 | Fishing Cat | ✅ | Vulnerable |
| 44 | Cheetah | ✅ | Vulnerable |
| 45 | Banteng | ✅ | Endangered |
| 46 | Indian Bison | ✅ | Vulnerable |

> **Note on reducing scope:** If you struggle to hit 46 classes with enough images (≥80/class), it's perfectly valid to use 35–40 classes. Report it honestly. Quality > quantity.

---

*Guide written for SMAI Assignment 3, IIIT Hyderabad AY 2025–26.*  
*All code is original and written for educational purposes.*  
*LLM assistance used for structuring and code scaffolding — disclose in report acknowledgements.*
