# Indian Mammal Classifier Team Brainiacs 


This project trains and serves a deep learning image classifier for a curated Indian mammal dataset using EfficientNet-B0, PyTorch, and Streamlit.

## Current Scope

This dataset consists of image data focused on Indian mammal species, curated specifically for image classification tasks.More information on dataset can be found in README.md file in data/ foler. Below are the categories:  

- Asiatic Lion
- Asiatic Wildcat
- Bengal Fox
- Chinkara
- Chital
- Four Horned Antelope
- Golden Jackal
- Honey Badger
- Indian Leopard
- Jungle Cat
- Nilgai
- Ruddy Mongoose
- Striped Hyena
- Wild Boar
- barasingha
- barking_deer
- bengal_tiger
- blackbuck
- bonnet_macaque
- desert_cat
- dhole
- ganges_river_dolphin
- gaur
- golden_langur
- greater_bandicoot_rat
- hanuman_langur
- himalayan_black_bear
- indian_elephant
- indian_flying_fox
- indian_giant_squirrel
- indian_grey_mongoose
- indian_hedgehog
- indian_palm_squirrel  
  
**link** - https://www.kaggle.com/datasets/prabhashpadhan/indianmammal  


## Project Structure

```text
IndianMammalClassifier/
├── app/
│   ├── app.py
│   ├── inference.py
│   └── species_info.json
├── data/
│   ├── README.md
├── models/
├── notebooks/
├── scripts/
│   ├── generate_metadata.py
│   ├── prepare_dataset.py
│   └── train.py
├── .gitignore
├── README.md
├── requirements.txt
└── T7_2_Indian_Mammals_Implementation_Guide.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 1. Prepare Train/Val Split

The source images should remain as class folders directly inside `data/`. Run:

```bash
python3 scripts/prepare_dataset.py
```

This creates reproducible `data/train/` and `data/val/` folders using an 80/20 split.

## 2. Train the Model

```bash
python3 scripts/train.py
```

Outputs:

- `models/indian_mammals_efficientnet_b0.pt`
- `models/class_names.json`
- `models/training_history.json`

## 3. Run the Streamlit App

```bash
streamlit run app/app.py
```

The app shows:

- uploaded image preview
- top-3 predictions
- confidence scores
- species information panel

## Notes

- This implementation is intentionally aligned to the curated mammal folders available under `data/`.
- `app/species_info.json` is generated from a local curated dictionary, so the app works without an external API.
- If you later add more mammal classes, update `scripts/prepare_dataset.py` and regenerate metadata if needed.
