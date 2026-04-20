from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from inference import (
    analyze_image,
    load_checkpoint,
    load_inference_config,
    load_mammal_detector,
    load_species_info,
)


st.set_page_config(page_title="Indian Mammal Classifier", layout="centered")


@st.cache_resource
def load_artifacts():
    model, classes, config = load_checkpoint()
    detector_model, detector_transform, mammal_indices = load_mammal_detector()
    inference_config = load_inference_config()
    species_info = load_species_info(Path("app/species_info.json"))
    return (
        model,
        classes,
        config,
        detector_model,
        detector_transform,
        mammal_indices,
        inference_config,
        species_info,
    )


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
        (
            model,
            classes,
            config,
            detector_model,
            detector_transform,
            mammal_indices,
            inference_config,
            species_info,
        ) = load_artifacts()
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

    analysis = analyze_image(
        image=image,
        classifier_model=model,
        classes=classes,
        classifier_config=config,
        detector_model=detector_model,
        detector_transform=detector_transform,
        mammal_indices=mammal_indices,
        inference_config=inference_config,
    )

    st.caption(f"Mammal filter score: {analysis['mammal_score'] * 100:.2f}%")

    if analysis["status"] == "rejected_non_mammal":
        st.error("Unknown / Not an Indian mammal")
        st.write(analysis["message"])
        return

    st.subheader("Top Predictions")
    for rank, pred in enumerate(analysis["predictions"], start=1):
        st.write(f"{rank}. {pred['label']} - {pred['confidence'] * 100:.2f}%")

    top_prediction = analysis["predictions"][0]
    st.progress(float(top_prediction["confidence"]))

    if analysis["status"] == "rejected_unknown":
        st.warning("Unknown / Not an Indian mammal")
        st.write(analysis["message"])
        return

    render_species_info(top_prediction["label"], species_info)


if __name__ == "__main__":
    main()
