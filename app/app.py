from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from inference import load_checkpoint, load_species_info, predict_image


st.set_page_config(page_title="Indian Mammal Classifier", layout="centered")


@st.cache_resource
def load_artifacts():
    model, classes, config = load_checkpoint()
    species_info = load_species_info(Path("app/species_info.json"))
    return model, classes, config, species_info


def render_species_info(label: str, species_info: dict) -> None:
    info = species_info.get(label)
    if not info:
        st.info("No metadata found for this species.")
        return

    st.subheader(label)
    st.write(info["summary"])
    st.markdown(f"**Scientific name:** {info['scientific_name']}")
    st.markdown(f"**IUCN status:** {info['iucn_status']}")
    st.markdown(f"**Habitat:** {info['habitat']}")
    st.markdown(f"**Distribution in India:** {info['distribution']}")
    st.markdown(f"**Diet:** {info['diet']}")
    st.markdown(f"**Fun fact:** {info['fun_fact']}")


def main() -> None:
    st.title("Indian Mammal Classifier")
    st.caption("EfficientNet-B0 image classifier for 14 Indian mammal classes.")

    try:
        model, classes, config, species_info = load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload an image of an Indian mammal", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is None:
        st.info("Upload an image to run inference.")
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    predictions = predict_image(
        image=image,
        model=model,
        classes=classes,
        img_size=config.get("img_size", 224),
        top_k=3,
    )

    st.subheader("Top Predictions")
    for rank, pred in enumerate(predictions, start=1):
        st.write(f"{rank}. {pred['label']} - {pred['confidence'] * 100:.2f}%")

    top_label = predictions[0]["label"]
    st.progress(float(predictions[0]["confidence"]))
    render_species_info(top_label, species_info)


if __name__ == "__main__":
    main()
