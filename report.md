# Indian Mammal Classifier Report

## 1. Project Overview

This project implements an image classification system for a curated Indian mammal dataset using:

- EfficientNet-B0 for supervised image classification
- PyTorch for training and inference
- Streamlit for the user-facing application
- Gemini 2.5 Flash for optional question answering in the app

The current implementation supports **30 mammal classes** and includes:

- dataset preparation and train/validation splitting
- model training with transfer learning
- saved checkpoint and class mapping
- automated training plots
- inference with top-3 predictions
- a mammal-vs-non-mammal rejection stage
- a species guide and chatbot layer in the UI

---

## 2. Repository Architecture

```text
IndianMammalClassifier/
├── app/
│   ├── app.py
│   ├── chatbot.py
│   ├── inference.py
│   └── species_info.json
├── config/
│   ├── inference_config.yaml
│   └── train_config.yaml
├── data/
│   ├── <source class folders>
│   ├── train/
│   └── val/
├── models/
│   ├── indian_mammals_efficientnet_b0.pt
│   ├── class_names.json
│   ├── training_history.json
│   └── plots/
├── scripts/
│   ├── prepare_dataset.py
│   ├── train.py
│   └── generate_metadata.py
├── README.md
└── report.md
```

---

## 3. Dataset Details

### 3.1 Supported Classes

The current curated dataset contains **30 mammal classes**:

1. Asiatic Lion
2. Asiatic Wildcat
3. Barasingha
4. Barking Deer
5. Bengal Fox
6. Bengal Tiger
7. Blackbuck
8. Bonnet Macaque
9. Chinkara
10. Chital
11. Dhole
12. Four Horned Antelope
13. Ganges River Dolphin
14. Gaur
15. Golden Jackal
16. Golden Langur
17. Greater Bandicoot Rat
18. Hanuman Langur
19. Honey Badger
20. Indian Elephant
21. Indian Flying Fox
22. Indian Giant Squirrel
23. Indian Grey Mongoose
24. Indian Leopard
25. Indian Palm Squirrel
26. Jungle Cat
27. Nilgai
28. Ruddy Mongoose
29. Striped Hyena
30. Wild Boar

### 3.2 Source Image Counts

Total curated source images: **5769**

| Class | Images |
|---|---:|
| Asiatic Lion | 126 |
| Asiatic Wildcat | 139 |
| Barasingha | 298 |
| Barking Deer | 299 |
| Bengal Fox | 100 |
| Bengal Tiger | 199 |
| Blackbuck | 295 |
| Bonnet Macaque | 299 |
| Chinkara | 106 |
| Chital | 104 |
| Dhole | 299 |
| Four Horned Antelope | 104 |
| Ganges River Dolphin | 120 |
| Gaur | 349 |
| Golden Jackal | 114 |
| Golden Langur | 97 |
| Greater Bandicoot Rat | 110 |
| Hanuman Langur | 299 |
| Honey Badger | 106 |
| Indian Elephant | 299 |
| Indian Flying Fox | 299 |
| Indian Giant Squirrel | 299 |
| Indian Grey Mongoose | 299 |
| Indian Leopard | 127 |
| Indian Palm Squirrel | 299 |
| Jungle Cat | 122 |
| Nilgai | 110 |
| Ruddy Mongoose | 139 |
| Striped Hyena | 107 |
| Wild Boar | 106 |

### 3.3 Excluded Folders

The following folders are currently skipped by the curated class map because they are too weak or were intentionally left out of the supported training set:

- desert_cat
- himalayan_black_bear
- indian_hedgehog

### 3.4 Train/Validation Split

The dataset preparation script performs a reproducible **80/20 split** using:

- random seed: `42`
- validation split: `0.20`
- minimum recommended images per class: `80`

Most recent split totals:

- Train images: **4631**
- Validation images: **1138**

---

## 4. Data Processing Pipeline

### 4.1 Dataset Preparation

Implemented in `scripts/prepare_dataset.py`.

Main steps:

1. Read class folders directly from `data/`
2. Map raw folder names to cleaned display names
3. Filter to the curated class catalog
4. Copy images into `data/train/<class_name>/` and `data/val/<class_name>/`
5. Preserve the original source folders

Supported file extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

### 4.2 Image Loading

The training loader uses a custom PIL loader that converts every image to RGB:

- `Image.open(path)`
- `image.convert("RGB")`

This avoids palette/transparency issues from mixed web-scraped image sources.

### 4.3 Training Transforms

Training augmentation:

- `RandomResizedCrop(224, scale=(0.7, 1.0))`
- `RandomHorizontalFlip()`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`
- `RandomRotation(10)`
- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

Validation/inference transform:

- `Resize(256)`
- `CenterCrop(224)`
- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

These mean/std values match ImageNet preprocessing and are consistent with pretrained EfficientNet usage.

---

## 5. Classification Model Architecture

### 5.1 Main Model

The primary classifier is:

- Model family: **EfficientNet**
- Variant: **EfficientNet-B0**
- Implementation: `timm.create_model("efficientnet_b0", pretrained=True, num_classes=30)`

### 5.2 Parameter Details

Exact model details for the current 30-class setup:

- Total parameters: **4,045,978**
- Trainable parameters: **4,045,978**
- Final classifier layer: `Linear(in_features=1280, out_features=30, bias=True)`

### 5.3 Why EfficientNet-B0

EfficientNet-B0 is a good fit here because it provides:

- low computational cost
- pretrained ImageNet transfer learning
- reasonable accuracy for medium-sized image datasets
- small enough footprint for deployment inside a lightweight Streamlit app

---

## 6. Training Configuration

Training configuration is stored in `config/train_config.yaml`.

Current settings:

- model name: `efficientnet_b0`
- image size: `224`
- batch size: `32`
- epochs: `30`
- learning rate (head): `0.001`
- learning rate (backbone): `0.0001`
- weight decay: `0.0001`
- label smoothing: `0.1`
- early stopping patience: `3`

### 6.1 Optimization Strategy

Optimizer:

- `AdamW`

Differential learning rates:

- backbone: `1e-4`
- classifier head: `1e-3`

Learning rate schedule:

- `CosineAnnealingLR`
- `eta_min = 1e-6`

Regularization:

- label smoothing
- data augmentation
- gradient clipping with `max_norm=1.0`

### 6.2 Data Loading

- training shuffle: enabled
- validation shuffle: disabled
- number of workers: `2`
- `pin_memory`: enabled when CUDA is available

---

## 7. Training Outputs

The training script saves:

- model checkpoint: `models/indian_mammals_efficientnet_b0.pt`
- class index mapping: `models/class_names.json`
- training history: `models/training_history.json`
- plots directory: `models/plots/`

### 7.1 Generated Plots

The following figures are generated automatically:

1. `training_curves.png`
   - train accuracy vs epoch
   - validation accuracy vs epoch
   - train loss vs epoch
   - validation loss vs epoch

2. `confusion_matrix.png`
   - confusion matrix on the validation set using the best saved checkpoint

---

## 8. Inference Pipeline

Inference is implemented in `app/inference.py`.

### 8.1 Stage 1: Mammal Filter

Before classifying into one of the supported species, the app runs a broad mammal-vs-non-mammal screening step.

Detector:

- architecture: **ResNet-18**
- source: `torchvision.models.resnet18`
- weights: `ResNet18_Weights.DEFAULT`

Method:

1. Run the uploaded image through pretrained ResNet-18
2. Convert logits to softmax probabilities
3. Sum probabilities over a manually curated set of mammal-related ImageNet categories
4. Compare the total mammal score against a threshold

Inference threshold from `config/inference_config.yaml`:

- `mammal_detector_threshold = 0.20`

If the mammal score is below this threshold, the app returns:

- `Unknown / Not an Indian mammal`

### 8.2 Stage 2: Fine-Grained Classifier

If the image passes the mammal filter, it is sent to the 30-class EfficientNet-B0 classifier.

Output:

- top-3 predictions using `torch.topk`
- softmax confidence values

Unknown-class rejection threshold:

- `classifier_unknown_threshold = 0.55`

If top-1 confidence is below this threshold, the app reports that the image is mammal-like but not confidently one of the supported Indian mammal classes.

### 8.3 Inference Output Parameters

From `config/inference_config.yaml`:

- mammal detector threshold: `0.20`
- classifier unknown threshold: `0.55`
- top-k predictions: `3`

---

## 9. Streamlit Application Architecture

The UI is implemented in `app/app.py`.

The interface is a three-panel dashboard:

1. **Left panel**
   - file uploader
   - uploaded image preview
   - top-3 predictions
   - confidence bars

2. **Center panel**
   - species guide
   - summary
   - scientific name
   - IUCN status
   - habitat
   - distribution
   - diet
   - fun fact

3. **Right panel**
   - chat interface
   - response source status
   - clear chat function

The app uses:

- `@st.cache_resource` for model and metadata loading
- custom CSS for dashboard styling
- `st.session_state` for chat history and chat status

---

## 10. Metadata and Knowledge Layer

Species metadata is generated by `scripts/generate_metadata.py` and stored in:

- `app/species_info.json`

Each species entry contains:

- scientific name
- IUCN status
- habitat
- distribution
- diet
- fun fact
- summary

Current metadata coverage:

- **30 species**

---

## 11. Chatbot Architecture

The chatbot logic is implemented in `app/chatbot.py`.

### 11.1 Model

Optional LLM backend:

- Gemini model: **gemini-2.5-flash**

### 11.2 Answering Strategy

The chatbot uses a hybrid approach:

1. **Dataset-first response**
   - If the question matches information already present in `species_info.json`, answer directly from the curated metadata.

2. **Gemini fallback**
   - If the answer is not present in local metadata, Gemini may answer from general biological knowledge.

3. **Local fallback**
   - If Gemini is unavailable or fails, the app returns a local summary-based response.

### 11.3 Environment Requirement

Gemini is enabled only if:

- `.env` exists
- `GEMINI_API_KEY` is set
- the `google-generativeai` package is installed

---

## 12. End-to-End System Flow

The full pipeline is:

1. User uploads image in Streamlit
2. Image is converted to RGB
3. Mammal filter checks whether the image is likely to contain a mammal
4. If accepted, EfficientNet-B0 predicts the top-3 mammal classes
5. Top class drives species-info retrieval from `species_info.json`
6. UI displays predictions, confidence, and species notes
7. User can ask questions in the chatbot panel
8. Chatbot answers from local metadata first, then Gemini if needed

---

## 13. Current Strengths

- Clean train/validation split pipeline
- Transfer learning from pretrained EfficientNet-B0
- Two-stage rejection logic for non-mammals and uncertain classes
- Curated species metadata integrated into the app
- Automated plot generation for training diagnostics
- Hybrid dataset-plus-LLM chatbot design

---

## 14. Current Limitations

- Class balance is uneven across species
- Some classes still have relatively small image counts compared with others
- Excluded mammal folders are not yet part of the curated training set
- Unknown-class rejection is threshold-based, not a fully learned open-set detector
- The chatbot does not use live web browsing and should not be treated as a live fact source

---

## 15. Reproducibility Commands

```bash
source venv/bin/activate
pip install -r requirements.txt
python3 scripts/generate_metadata.py
python3 scripts/prepare_dataset.py
python3 scripts/train.py
streamlit run app/app.py
```

---

## 16. Summary

This project currently implements a **30-class Indian mammal recognition system** with:

- curated dataset preparation
- EfficientNet-B0 fine-tuning
- top-3 probability-based inference
- non-mammal rejection
- confidence-based unknown handling
- species metadata lookup
- optional Gemini-assisted question answering

The architecture is lightweight, practical, and suitable for an academic image-classification project with both model-training and interactive application components.
